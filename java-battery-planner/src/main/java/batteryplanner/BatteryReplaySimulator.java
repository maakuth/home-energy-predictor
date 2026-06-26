package batteryplanner;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * Replay simulator that replays historical data (fixtures) through a
 * {@link BatteryPlanner} to evaluate its performance.
 *
 * <h3>How the replay loop works</h3>
 * <p>The simulator replays measurement data interval by interval (15 min).
 * At each step it:
 *
 * <ol>
 *   <li>Determines the current simulated time.</li>
 *   <li>Builds the <em>planner horizon</em> — the predictions, prices, and
 *       timestamps that would have been visible <strong>to a planner running
 *       at that moment in real life</strong>.  This uses the
 *       {@code predictions_archive} from the fixture, selecting only
 *       prediction records whose {@code generated_at} ≤ current time
 *       and {@code target_timestamp} ≥ current time.</li>
 *   <li>Calls the planner's {@link BatteryPlanner#plan plan()} with the
 *       horizon data and the current battery SoC.</li>
 *   <li>Takes only the <strong>first interval</strong> of the plan
 *       (receding-horizon / model-predictive control pattern).</li>
 *   <li>Applies the planned battery charge/discharge to the
 *       <strong>actual</strong> measured load of that interval
 *       (not the predicted load), accumulating real cost.</li>
 *   <li>Advances time and loops.</li>
 * </ol>
 *
 * <p>The key design point: the planner sees only what it would have seen
 * in production (predictions generated before the current time), and the
 * cost is computed using <em>actual</em> measurements — so the replay is
 * a faithful historical backtest.
 *
 * <h3>Handling of time in the Java side</h3>
 * <p>The replay produces a time-ordered sequence of plans.  At each step
 * the planner receives the full horizon visible from that point forward,
 * but only its first-entry decision is applied.  This means a planner
 * <strong>does not need to track time itself</strong> — it receives
 * whatever horizon is currently visible and returns one entry per interval.
 *
 * <h3>Adding a new planner</h3>
 * <ol>
 *   <li>Implement {@link BatteryPlanner}.</li>
 *   <li>Add it to {@link BatteryPlannerReplayTest#discoverPlanners
 *       discoverPlanners()} so all fixture combinations are tested.</li>
 *   <li>The test class will run the replay automatically, reporting
 *       savings %, SoC violations, and cost metrics.</li>
 * </ol>
 */
public class BatteryReplaySimulator {

    private static final double INTERVAL_HOURS = 0.25;
    private static final DateTimeFormatter ISO_FMT = DateTimeFormatter.ISO_OFFSET_DATE_TIME;

    private final FixtureData fixture;
    private final List<MeasurementRecord> measurements;
    private final List<ArchiveRecord> archive;

    public BatteryReplaySimulator(FixtureData fixture) {
        this.fixture = fixture;
        this.measurements = parseMeasurements(fixture.getMeasurements());
        this.archive = parseArchive(fixture.getPredictionsArchive());
    }

    public FixtureData getFixture() { return fixture; }
    public List<MeasurementRecord> getMeasurements() { return measurements; }
    public boolean hasMeasurements() { return !measurements.isEmpty(); }
    public boolean hasArchive() { return !archive.isEmpty(); }

    /**
     * Run a full replay of the planner against the fixture data.
     *
     * @param planner     the planner to evaluate
     * @param plannerType a human-readable name for logging
     * @param config      battery parameters
     * @param maxPlanks   maximum number of intervals to simulate
     * @return a {@link ReplayResult} with cost, violations, savings
     */
    public ReplayResult simulate(
            BatteryPlanner planner, String plannerType,
            BatteryConfig config, int maxPlanks) {

        ReplayResult result = new ReplayResult();
        result.setPlannerType(plannerType);

        if (!hasMeasurements()) {
            result.setSuccess(false);
            result.setError("No measurement data in fixture");
            return result;
        }

        double cap = config.getCapacityKwh();
        double minSocPct = config.getMinSocPct();
        double maxSocPct = config.getMaxSocPct();
        double minSocKwh = cap * minSocPct / 100.0;
        double maxSocKwh = cap * maxSocPct / 100.0;
        double socKwh = cap * config.getInitialSocPct() / 100.0;

        OffsetDateTime currentTime = measurements.get(0).timestamp;
        OffsetDateTime endTime = measurements.get(measurements.size() - 1).timestamp;

        int intervalsRun = 0;
        List<Map<String, Object>> socViolations = new ArrayList<>();
        double costWithBattery = 0.0;
        double costNoBattery = 0.0;

        while (currentTime.compareTo(endTime) <= 0 && intervalsRun < maxPlanks) {
            double currentSocPct = (socKwh / cap) * 100.0;

            double[] predictions;
            double[] solar;
            double[] importPrices;
            double[] exportPrices;
            List<String> timestamps;

            try {
                PlannerHorizonData horizon = getPlannerHorizon(currentTime, maxPlanks);
                predictions = horizon.predictionsKwh;
                solar = horizon.solarKwh;
                importPrices = horizon.importPrices;
                exportPrices = horizon.exportPrices;
                timestamps = horizon.timestamps;
            } catch (Exception e) {
                result.setSuccess(false);
                result.setError("getPlannerHorizon: " + e.getMessage());
                result.setIntervalsRun(intervalsRun);
                return result;
            }

            if (predictions.length == 0) {
                break;
            }

            try {
                List<BatteryPlanEntry> plan = planner.plan(
                    predictions, solar, importPrices, exportPrices,
                    timestamps, null, true, currentSocPct, null);

                if (plan == null || plan.isEmpty()) {
                    break;
                }

                BatteryPlanEntry entry = plan.get(0);

                MeasurementRecord meas = findMeasurement(currentTime);
                double actualSolar = 0.0;
                double actualLoadKw = 0.0;
                if (meas != null) {
                    actualSolar = meas.solarActualKw;
                    actualLoadKw = meas.totalPowerKw + actualSolar;
                }
                double actualLoadKwh = actualLoadKw * INTERVAL_HOURS;

                double batChargeKwh = entry.getChargeFromSolarKwh() + entry.getChargeFromGridKwh();
                double batDischargeKwh = entry.getDischargeToLoadKwh() + entry.getDischargeToExportKwh();

                socKwh = socKwh + batChargeKwh - batDischargeKwh;
                socKwh = clamp(socKwh, minSocKwh, maxSocKwh);

                double socAfterPct = (socKwh / cap) * 100.0;
                if (socAfterPct < minSocPct || socAfterPct > maxSocPct) {
                    Map<String, Object> v = new LinkedHashMap<>();
                    v.put("timestamp", currentTime.format(ISO_FMT));
                    v.put("socPct", socAfterPct);
                    v.put("min", minSocPct);
                    v.put("max", maxSocPct);
                    socViolations.add(v);
                }

                double importPrice = importPrices.length > 0 ? importPrices[0] : 0.15;
                double exportPrice = exportPrices.length > 0 ? exportPrices[0] : 0.05;

                double gridImport = Math.max(0.0, actualLoadKwh + batChargeKwh - batDischargeKwh);
                double gridExport = Math.max(0.0, batDischargeKwh - actualLoadKwh - batChargeKwh);

                costWithBattery += gridImport * importPrice - gridExport * exportPrice;
                costNoBattery += actualLoadKwh * importPrice;

            } catch (Exception e) {
                result.setSuccess(false);
                result.setError("Planner error at " + currentTime.format(ISO_FMT) + ": " + e.getMessage());
                result.setIntervalsRun(intervalsRun);
                return result;
            }

            currentTime = currentTime.plusMinutes(15);
            intervalsRun++;
        }

        double savings = costNoBattery - costWithBattery;

        result.setSuccess(true);
        result.setIntervalsRun(intervalsRun);
        result.setSocViolations(socViolations.size());
        result.setSocViolationDetails(socViolations);
        result.setCostWithBatteryEur(costWithBattery);
        result.setCostNoBatteryEur(costNoBattery);
        result.setSavingsEur(savings);
        result.setSavingsPct(costNoBattery > 0 ? (savings / costNoBattery * 100.0) : 0.0);
        result.setFinalSocPct((socKwh / cap) * 100.0);

        return result;
    }

    /**
     * Builds the forecast horizon visible to the planner at {@code planningTime}.
     * Selects predictions from the archive that were generated before or at
     * the planning time and target at or after it, keeping only the most
     * recent forecast for each target timestamp.
     */
    private PlannerHorizonData getPlannerHorizon(OffsetDateTime planningTime, int length) {
        Map<OffsetDateTime, ArchiveRecord> visibleByTarget = getVisiblePredictions(planningTime);

        if (visibleByTarget.isEmpty()) {
            return new PlannerHorizonData(new double[0], new double[0], new double[0], new double[0], new ArrayList<>());
        }

        List<OffsetDateTime> sortedTargets = new ArrayList<>(visibleByTarget.keySet());
        Collections.sort(sortedTargets);

        int n = Math.min(sortedTargets.size(), length);
        double[] pred = new double[n];
        double[] solar = new double[n];
        double[] imp = new double[n];
        double[] exp = new double[n];
        List<String> ts = new ArrayList<>(n);

        for (int i = 0; i < n; i++) {
            OffsetDateTime tgt = sortedTargets.get(i);
            ArchiveRecord rec = visibleByTarget.get(tgt);
            pred[i] = rec.predictedUsageKw * INTERVAL_HOURS;
            solar[i] = rec.solarForecastKw * INTERVAL_HOURS;
            imp[i] = rec.importPrice;
            exp[i] = rec.exportPrice;
            ts.add(tgt.format(ISO_FMT));
        }

        return new PlannerHorizonData(pred, solar, imp, exp, ts);
    }

    /**
     * Filters the archive to predictions that were visible at {@code planningTime} —
     * i.e., generated at or before planningTime AND targeting at or after it.
     * When multiple predictions exist for the same target timestamp, the most
     * recently generated one wins.
     */
    private Map<OffsetDateTime, ArchiveRecord> getVisiblePredictions(OffsetDateTime planningTime) {
        Map<OffsetDateTime, ArchiveRecord> latestPerTarget = new TreeMap<>();

        for (ArchiveRecord rec : archive) {
            if (!rec.generatedAt.isAfter(planningTime) && !rec.targetTimestamp.isBefore(planningTime)) {
                ArchiveRecord existing = latestPerTarget.get(rec.targetTimestamp);
                if (existing == null || rec.generatedAt.isAfter(existing.generatedAt)) {
                    latestPerTarget.put(rec.targetTimestamp, rec);
                }
            }
        }

        return latestPerTarget;
    }

    private MeasurementRecord findMeasurement(OffsetDateTime timestamp) {
        OffsetDateTime rounded = timestamp.truncatedTo(ChronoUnit.MINUTES)
                .withMinute((timestamp.getMinute() / 15) * 15);
        int idx = Collections.binarySearch(measurements, null, (a, b) -> {
            OffsetDateTime t = rounded;
            return a.timestamp.compareTo(t);
        });
        if (idx >= 0) return measurements.get(idx);
        int insertIdx = -(idx + 1);
        if (insertIdx > 0) return measurements.get(insertIdx - 1);
        return null;
    }

    private List<MeasurementRecord> parseMeasurements(List<Map<String, Object>> raw) {
        List<MeasurementRecord> result = new ArrayList<>();
        for (Map<String, Object> m : raw) {
            try {
                OffsetDateTime ts = parseInstant(m.get("timestamp"));
                double totalPower = doubleVal(m.get("total_power_kw"));
                double solarActual = doubleVal(m.get("solar_actual_kw"));
                double gshpPower = doubleVal(m.get("gshp_power_kw"));
                double outsideTemp = doubleVal(m.get("outside_temp_c"));
                result.add(new MeasurementRecord(ts, totalPower, solarActual, gshpPower, outsideTemp));
            } catch (Exception ignored) {}
        }
        result.sort(Comparator.comparing(r -> r.timestamp));
        return result;
    }

    private List<ArchiveRecord> parseArchive(List<Map<String, Object>> raw) {
        List<ArchiveRecord> result = new ArrayList<>();
        for (Map<String, Object> m : raw) {
            try {
                OffsetDateTime target = parseInstant(m.get("target_timestamp"));
                OffsetDateTime generated = parseInstant(m.get("generated_at"));
                double predictedUsage = doubleVal(m.get("predicted_usage_kw"));
                double solarForecast = doubleVal(m.get("solar_forecast_kw"));
                double importPrice = doubleVal(m.get("import_price"));
                double exportPrice = doubleVal(m.get("export_price"));
                result.add(new ArchiveRecord(target, generated, predictedUsage,
                                             solarForecast, importPrice, exportPrice));
            } catch (Exception ignored) {}
        }
        return result;
    }

    private static OffsetDateTime parseInstant(Object v) {
        if (v == null) throw new IllegalArgumentException("null timestamp");
        String s = v.toString().replace(" ", "T");
        return OffsetDateTime.parse(s, ISO_FMT);
    }

    private static double doubleVal(Object v) {
        if (v == null) return 0.0;
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); }
        catch (NumberFormatException e) { return 0.0; }
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /** A single measurement record from the fixture (actuals). */
    public static class MeasurementRecord {
        public final OffsetDateTime timestamp;
        public final double totalPowerKw;
        public final double solarActualKw;
        public final double gshpPowerKw;
        public final double outsideTempC;

        public MeasurementRecord(OffsetDateTime timestamp, double totalPowerKw,
                                 double solarActualKw, double gshpPowerKw, double outsideTempC) {
            this.timestamp = timestamp;
            this.totalPowerKw = totalPowerKw;
            this.solarActualKw = solarActualKw;
            this.gshpPowerKw = gshpPowerKw;
            this.outsideTempC = outsideTempC;
        }
    }

    /** A single prediction archive record — one forecast point (15 min). */
    public static class ArchiveRecord {
        public final OffsetDateTime targetTimestamp;
        public final OffsetDateTime generatedAt;
        public final double predictedUsageKw;
        public final double solarForecastKw;
        public final double importPrice;
        public final double exportPrice;

        public ArchiveRecord(OffsetDateTime targetTimestamp, OffsetDateTime generatedAt,
                             double predictedUsageKw, double solarForecastKw,
                             double importPrice, double exportPrice) {
            this.targetTimestamp = targetTimestamp;
            this.generatedAt = generatedAt;
            this.predictedUsageKw = predictedUsageKw;
            this.solarForecastKw = solarForecastKw;
            this.importPrice = importPrice;
            this.exportPrice = exportPrice;
        }
    }

    private static class PlannerHorizonData {
        final double[] predictionsKwh;
        final double[] solarKwh;
        final double[] importPrices;
        final double[] exportPrices;
        final List<String> timestamps;

        PlannerHorizonData(double[] predictionsKwh, double[] solarKwh,
                           double[] importPrices, double[] exportPrices,
                           List<String> timestamps) {
            this.predictionsKwh = predictionsKwh;
            this.solarKwh = solarKwh;
            this.importPrices = importPrices;
            this.exportPrices = exportPrices;
            this.timestamps = timestamps;
        }
    }
}
