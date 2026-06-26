package batteryplanner;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A rule-based (heuristic) battery planner that serves as both the
 * production fallback and a reference implementation for new planners.
 *
 * <h3>Algorithm overview</h3>
 * <p>For each interval (in order) the planner evaluates four dispatch
 * decisions — it performs the <em>first</em> action whose conditions are
 * met, then moves to the next interval:
 *
 * <ol>
 *   <li><b>Charge from solar surplus</b> — if net load (load − solar) is
 *       negative, there is solar excess.  Charge the battery unless it is
 *       more profitable to export the surplus (compares current export
 *       price with the opportunity cost of round-trip storage).</li>
 *   <li><b>Discharge to load</b> — if net load is positive and the current
 *       import price is at or above the maximum remaining price, use battery
 *       energy to serve the load.  This implements <em>"peak shaving"</em>:
 *       discharge during the most expensive intervals.</li>
 *   <li><b>Discharge to export</b> — if spare discharge capacity remains
 *       and export is allowed, sell energy to the grid when either the
 *       current export price is the best remaining or there is arbitrage
 *       profit (export price &gt; future lowest import / round-trip eff).</li>
 *   <li><b>Charge from grid</b> — at the cheapest import price window,
 *       charge the battery if the round trip is profitable AND there isn't
 *       enough expected future solar to fill the battery for free anyway.</li>
 * </ol>
 *
 * <p><b>Limitations (good to know when writing a new planner):</b>
 * <ul>
 *   <li>The greedy per-interval logic cannot globally optimise across
 *       the entire horizon.  A linear-programming approach would.</li>
 *   <li>The context map is currently ignored — outside temperature,
 *       EV position, SARIMA bounds, etc. are available but unused.</li>
 *   <li>There is a known bug where the greedy grid-charge decision
 *       (step 4) is skipped when any other action was taken, even if the
 *       battery could still accept more energy at a cheap price.</li>
 * </ul>
 *
 * @see BatteryPlanner
 * @see BatteryConfig
 */
public class HeuristicBatteryPlanner implements BatteryPlanner {

    private final BatteryConfig config;

    public HeuristicBatteryPlanner() {
        this(new BatteryConfig());
    }

    public HeuristicBatteryPlanner(BatteryConfig config) {
        this.config = config;
    }

    @Override
    public List<BatteryPlanEntry> plan(
            double[] predictionsKwh,
            double[] solarKwh,
            double[] importPrices,
            double[] exportPrices,
            List<String> predictionTimestamps,
            double[] committedLoadKwh,
            boolean allowExport,
            Double initialSocPct,
            Map<String, Object> context) {

        int horizon = predictionsKwh.length;
        double intervalHours = 0.25;

        double cap = config.getCapacityKwh();
        double minSocKwh = config.getMinSocKwh();
        double maxSocKwh = config.getMaxSocKwh();
        double maxChargeKw = config.getMaxChargeKw();
        double maxDischargeKw = config.getMaxDischargeKw();
        double chargeEff = config.getChargeEfficiency();
        double dischargeEff = config.getDischargeEfficiency();

        double socPct = initialSocPct != null ? initialSocPct : config.getInitialSocPct();
        double socKwh = cap * socPct / 100.0;
        socKwh = clamp(socKwh, minSocKwh, maxSocKwh);

        double[] committed = committedLoadKwh != null ? committedLoadKwh : new double[horizon];
        double[] netWithoutBattery = new double[horizon];
        for (int i = 0; i < horizon; i++) {
            netWithoutBattery[i] = predictionsKwh[i] - solarKwh[i];
        }

        List<BatteryPlanEntry> plan = new ArrayList<>(horizon);

        for (int i = 0; i < horizon; i++) {
            double netLoad = netWithoutBattery[i];
            double currentImport = importPrices[i];
            double currentExport = exportPrices[i];
            double futureMaxImport = maxRemaining(importPrices, i + 1, currentImport);
            double futureMinImport = minRemaining(importPrices, i + 1, currentImport);

            double chargeFromSolar = 0.0;
            double chargeFromGrid = 0.0;
            double dischargeToLoad = 0.0;
            double dischargeToExport = 0.0;

            // --- 1. Charge from solar surplus ---
            if (netLoad < 0) {
                double solarSurplus = -netLoad;
                double socRoomKwh = maxSocKwh - socKwh;
                double chargeLimitInput = Math.min(
                    maxChargeKw * intervalHours,
                    socRoomKwh / chargeEff);

                double roundTripEff = chargeEff * dischargeEff;
                double oppCost = futureMaxImport * roundTripEff;
                boolean exportBetter = allowExport && currentExport > oppCost;

                if (!exportBetter) {
                    chargeFromSolar = Math.min(solarSurplus, chargeLimitInput);
                    socKwh += chargeFromSolar * chargeEff;
                }
            }

            // --- 2. Discharge to load (peak shaving) ---
            if (netLoad > 0) {
                double socAvailableKwh = socKwh - minSocKwh;
                double dischargeLimitKwh = Math.min(
                    maxDischargeKw * intervalHours,
                    socAvailableKwh * dischargeEff);

                boolean shouldDischarge;
                if (futureMaxImport > 0) {
                    shouldDischarge = currentImport >= futureMaxImport;
                } else {
                    shouldDischarge = currentImport >= futureMaxImport;
                }

                if (shouldDischarge) {
                    dischargeToLoad = Math.min(netLoad, dischargeLimitKwh);
                    socKwh -= dischargeToLoad / dischargeEff;
                }
            }

            // --- 3. Discharge to export (arbitrage) ---
            if (allowExport && chargeFromSolar == 0.0) {
                double socAvailableKwh = socKwh - minSocKwh;
                double totalDischargeLimit = Math.min(
                    maxDischargeKw * intervalHours,
                    socAvailableKwh * dischargeEff);
                double remainingCap = Math.max(0.0, totalDischargeLimit - dischargeToLoad);

                if (remainingCap > 0) {
                    boolean isBestExport = currentExport >= maxRemaining(exportPrices, i + 1, currentExport);
                    double roundTripEff = chargeEff * dischargeEff;
                    boolean isArbitrage = currentExport > (futureMinImport / roundTripEff);

                    if (isBestExport || isArbitrage) {
                        dischargeToExport = Math.min(remainingCap,
                            socAvailableKwh * dischargeEff - dischargeToLoad);
                        dischargeToExport = Math.max(0.0, dischargeToExport);
                        socKwh -= dischargeToExport / dischargeEff;
                    }
                }
            }

            // --- 4. Charge from grid (cheapest price window) ---
            if (netLoad <= 0 && chargeFromSolar == 0.0 && dischargeToLoad == 0.0 && dischargeToExport == 0.0) {
                double lookaheadSteps = Math.min(96, horizon - i - 1);
                double expectedSolarSurplusKwh = 0.0;
                for (int j = 1; j <= lookaheadSteps; j++) {
                    if (netWithoutBattery[i + j] < 0) {
                        expectedSolarSurplusKwh += -netWithoutBattery[i + j];
                    }
                }

                double remainingRoomKwh = maxSocKwh - socKwh;
                boolean solarCanFill = expectedSolarSurplusKwh >= (remainingRoomKwh / chargeEff);

                boolean profitableGridCharge = (futureMaxImport * chargeEff) > currentImport;
                boolean isCheapestWindow = currentImport <= futureMinImport + 1e-9;

                if (profitableGridCharge && !solarCanFill && isCheapestWindow) {
                    double committedVal = committed[i];
                    double existingGridImport = Math.max(
                        netLoad + committedVal - dischargeToLoad - dischargeToExport, 0.0);

                    double socRoomKwh = maxSocKwh - socKwh;
                    double chargeLimitInput = Math.min(
                        maxChargeKw * intervalHours,
                        socRoomKwh / chargeEff);

                    chargeFromGrid = Math.min(chargeLimitInput, chargeLimitInput);
                    socKwh += chargeFromGrid * chargeEff;
                }
            }

            socKwh = clamp(socKwh, minSocKwh, maxSocKwh);

            // --- Build entry ---
            double totalCharge = chargeFromSolar + chargeFromGrid;
            double totalDischarge = dischargeToLoad + dischargeToExport;
            double batPowerKw = (totalCharge - totalDischarge) / intervalHours;

            double committedVal = committed[i];
            double totalNetAfterBat = netLoad + totalCharge - totalDischarge + committedVal;
            double gridImportKwh = Math.max(totalNetAfterBat, 0.0);
            double gridExportKwh = Math.max(-totalNetAfterBat, 0.0);

            double noBatImport = Math.max(netLoad + committedVal, 0.0);
            double noBatExport = Math.max(-(netLoad + committedVal), 0.0);
            double costNoBat = (noBatImport * currentImport) - (noBatExport * currentExport);
            double costWithBat = (gridImportKwh * currentImport) - (gridExportKwh * currentExport);

            String action;
            if (chargeFromSolar > 1e-9 && chargeFromGrid > 1e-9)
                action = "charge_mixed";
            else if (chargeFromSolar > 1e-9)
                action = "charge_solar";
            else if (chargeFromGrid > 1e-9)
                action = "charge_grid";
            else if (dischargeToLoad > 1e-9 && dischargeToExport > 1e-9)
                action = "discharge_mixed";
            else if (dischargeToLoad > 1e-9)
                action = "discharge_load";
            else if (dischargeToExport > 1e-9)
                action = "discharge_export";
            else
                action = "idle";

            String ts = i < predictionTimestamps.size() ? predictionTimestamps.get(i)
                        : String.format("T+%d", i);

            double socPctFinal = cap > 0 ? (socKwh / cap) * 100.0 : 0.0;

            plan.add(new BatteryPlanEntry(
                ts, action, batPowerKw,
                chargeFromSolar, chargeFromGrid,
                dischargeToLoad, dischargeToExport,
                socKwh, socPctFinal,
                gridImportKwh, gridExportKwh,
                costWithBat, costNoBat - costWithBat,
                netLoad
            ));
        }

        return plan;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double maxRemaining(double[] arr, int start, double fallback) {
        if (start >= arr.length) return fallback;
        double m = arr[start];
        for (int i = start + 1; i < arr.length; i++) {
            if (arr[i] > m) m = arr[i];
        }
        return m;
    }

    private static double minRemaining(double[] arr, int start, double fallback) {
        if (start >= arr.length) return fallback;
        double m = arr[start];
        for (int i = start + 1; i < arr.length; i++) {
            if (arr[i] < m) m = arr[i];
        }
        return m;
    }
}
