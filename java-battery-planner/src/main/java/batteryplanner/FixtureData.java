package batteryplanner;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Container for fixture data loaded from a JSON fixture file.
 *
 * <p>A fixture captures a snapshot of the system state at a point in time:
 * <ul>
 *   <li>{@code metadata} — fixture period, model version, etc.</li>
 *   <li>{@code ha_states} — current Home Assistant entity states</li>
 *   <li>{@code predictions} — the most recent load/solar forecasts</li>
 *   <li>{@code market_prices} — day-ahead spot prices</li>
 *   <li>{@code measurements} — historical <em>actual</em> consumption
 *       (used as ground truth in replay)</li>
 *   <li>{@code predictions_archive} — all forecast snapshots that were
 *       generated over time, enabling faithful time-aware replay</li>
 *   <li>{@code battery_config} — battery physical parameters</li>
 *   <li>{@code gshp_config} — GSHP configuration</li>
 * </ul>
 *
 * <p>The {@link BatteryReplaySimulator} uses the archive + measurements
 * to reconstruct what a planner would have seen at each point in time.
 */
public class FixtureData {

    private Map<String, Object> raw;

    public FixtureData() {}

    public FixtureData(Map<String, Object> raw) {
        this.raw = raw;
        parse();
    }

    public void setRaw(Map<String, Object> raw) {
        this.raw = raw;
        parse();
    }

    private Map<String, Object> metadata = new LinkedHashMap<>();
    private Map<String, Object> haStates = new LinkedHashMap<>();
    private List<Map<String, Object>> predictions = new ArrayList<>();
    private List<Map<String, Object>> marketPrices = new ArrayList<>();
    private List<Map<String, Object>> measurements = new ArrayList<>();
    private List<Map<String, Object>> predictionsArchive = new ArrayList<>();
    private Map<String, Object> batteryConfig = new LinkedHashMap<>();
    private Map<String, Object> gshpConfig = new LinkedHashMap<>();

    @SuppressWarnings("unchecked")
    private void parse() {
        if (raw == null) return;
        metadata = safeMap(raw.get("metadata"));
        haStates = safeMap(raw.get("ha_states"));
        predictions = safeList(raw.get("predictions"));
        marketPrices = safeList(raw.get("market_prices"));
        batteryConfig = safeMap(raw.get("battery_config"));
        gshpConfig = safeMap(raw.get("gshp_config"));

        Map<String, Object> history = safeMap(raw.get("history"));
        measurements = safeList(history.get("measurements"));
        predictionsArchive = safeList(history.get("predictions_archive"));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> safeMap(Object o) {
        if (o instanceof Map) return (Map<String, Object>) o;
        return new LinkedHashMap<>();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> safeList(Object o) {
        if (o instanceof List) {
            List<?> l = (List<?>) o;
            List<Map<String, Object>> result = new ArrayList<>();
            for (Object item : l) {
                if (item instanceof Map) {
                    result.add((Map<String, Object>) item);
                }
            }
            return result;
        }
        return new ArrayList<>();
    }

    public Map<String, Object> getMetadata() { return metadata; }
    public Map<String, Object> getHaStates() { return haStates; }
    public List<Map<String, Object>> getPredictions() { return predictions; }
    public List<Map<String, Object>> getMarketPrices() { return marketPrices; }
    public List<Map<String, Object>> getMeasurements() { return measurements; }
    public List<Map<String, Object>> getPredictionsArchive() { return predictionsArchive; }
    public Map<String, Object> getBatteryConfig() { return batteryConfig; }
    public Map<String, Object> getGshpConfig() { return gshpConfig; }

    public String getModelVersion() {
        Object v = metadata.get("model_version");
        return v != null ? v.toString() : "unknown";
    }

    public Instant getPeriodStart() {
        Object v = metadata.get("period_start");
        return v != null ? Instant.parse(v.toString().replace(" ", "T")) : null;
    }

    public Instant getPeriodEnd() {
        Object v = metadata.get("period_end");
        return v != null ? Instant.parse(v.toString().replace(" ", "T")) : null;
    }

    public String getSummary() {
        return String.format(
            "Fixture: period=%s..%s, model=%s, archive=%d, measurements=%d",
            getPeriodStart(), getPeriodEnd(), getModelVersion(),
            predictionsArchive.size(), measurements.size());
    }
}
