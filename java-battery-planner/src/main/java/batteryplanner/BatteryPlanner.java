package batteryplanner;

import java.util.List;
import java.util.Map;

/**
 * Interface for battery dispatch planning strategies.
 *
 * <p>A battery planner receives load forecasts, solar forecasts, and spot-price
 * predictions, then produces a per-interval dispatch plan (list of
 * {@link BatteryPlanEntry}) that optimises battery usage according to the
 * planner's strategy — heuristic, optimisation (LP/MILP), RL-based, etc.
 *
 * <h3>Contract</h3>
 * <ul>
 *   <li>All array arguments have the same length, equal to the planning
 *       horizon (number of 15-minute intervals).</li>
 *   <li>Energy values are <b>kWh per interval</b> (not kW).
 *       Because intervals are 15 minutes, multiply kW by 0.25 to convert.</li>
 *   <li>Prices are in EUR/kWh (including VAT, transfer fees, etc.).</li>
 *   <li>The planner <b>must</b> return exactly one {@link BatteryPlanEntry}
 *       per input interval (i.e. {@code plan.size() == predictionsKwh.length}).</li>
 *   <li>Every entry must have finite (non-NaN, non-Infinite) values for
 *       {@code batteryPowerKw}, {@code socKwh}/{@code socPct},
 *       {@code gridImportKwh}, {@code gridExportKwh}.</li>
 *   <li>The planner must not let the simulated SoC drop below 0 % or exceed
 *       100 %.  It is recommended to respect the min/max SoC bounds from
 *       {@link BatteryConfig} via environment or the {@code context} map.</li>
 *   <li>The planner should treat {@code committedLoadKwh} as load that
 *       consumes grid capacity but is <em>not</em> served by the house
 *       battery (e.g. an EV charger behind the main fuse).</li>
 * </ul>
 *
 * <h3>Context map</h3>
 * <p>The optional {@code context} map provides extra data that some planners
 * can use for better decisions.  All keys are optional; a planner must
 * silently ignore keys it does not recognise so the protocol can be extended.
 * Standard keys:
 *
 * <table border="1">
 *   <caption>Standard context keys</caption>
 *   <tr><th>Key</th><th>Type</th><th>Description</th></tr>
 *   <tr><td>{@code "outside_temps"}</td>
 *       <td>{@code double[]}</td>
 *       <td>Outside air temperature (°C) per interval.  Useful for
 *           anticipating heating-load changes.</td></tr>
 *   <tr><td>{@code "is_sauna_active"}</td>
 *       <td>{@code int[]} (0/1)</td>
 *       <td>Binary flag — the sauna is expected to be on during each
 *           interval.  Hot-water demand spikes can shift optimal discharge
 *           timing.</td></tr>
 *   <tr><td>{@code "ev_position"}</td>
 *       <td>{@code int[]} (0/1)</td>
 *       <td>{@code 1} means the EV is at home.  When the EV is away its
 *           committed charging load disappears, freeing grid capacity.</td></tr>
 *   <tr><td>{@code "sarima_lower"}</td>
 *       <td>{@code double[]}</td>
 *       <td>95 % lower prediction bound for baseload (kW).  Enables
 *           risk-aware / robust planning.</td></tr>
 *   <tr><td>{@code "sarima_upper"}</td>
 *       <td>{@code double[]}</td>
 *       <td>95 % upper prediction bound for baseload (kW).</td></tr>
 *   <tr><td>{@code "is_fallback_price"}</td>
 *       <td>{@code int[]} (0/1)</td>
 *       <td>{@code 1} when the price is a fallback estimate rather than a
 *           real day-ahead spot price.  A cautious planner might avoid
 *           aggressive arbitrage on fallback prices.</td></tr>
 *   <tr><td>{@code "tomorrow_valid"}</td>
 *       <td>{@code Boolean}</td>
 *       <td>{@code true} when tomorrow's day-ahead prices are published
 *           (after ~15:00 local time).  A planner can cap its lookahead
 *           when this is {@code false}.</td></tr>
 *   <tr><td>{@code "planned_gshp_kw"}</td>
 *       <td>{@code double[]}</td>
 *       <td>Planned GSHP electric load (kW).  Separated from baseload so the
 *           planner can model heat-pump ramping constraints.</td></tr>
 *   <tr><td>{@code "current_acc_temp"}</td>
 *       <td>{@code Double}</td>
 *       <td>Current accumulator temperature (°C).  Relevant for
 *           co-optimisation with thermal storage.</td></tr>
 *   <tr><td>{@code "is_fireplace_currently_on"}</td>
 *       <td>{@code Boolean}</td>
 *       <td>{@code true} when the fireplace is actively heating the
 *           accumulator.  This reduces expected GSHP load and may create
 *           extra margin for battery discharge.</td></tr>
 *   <tr><td>{@code "model_version"}</td>
 *       <td>{@code String}</td>
 *       <td>Semantic version string (e.g. {@code "1.2.0"}) from the model
 *           version file.  Useful for tagging experiments.</td></tr>
 * </table>
 *
 * <h3>How to implement</h3>
 * <ol>
 *   <li>Write a class that implements {@code BatteryPlanner}.</li>
 *   <li>Register it in {@code discoverPlanners()} inside
 *       {@code BatteryPlannerReplayTest.java} so it is tested.</li>
 *   <li>The {@link HeuristicBatteryPlanner} is a complete working example
 *       — use it as a starting template.</li>
 * </ol>
 *
 * @see HeuristicBatteryPlanner
 * @see BatteryConfig
 * @see BatteryPlanEntry
 */
public interface BatteryPlanner {

    /**
     * Generate a battery dispatch plan.
     *
     * @param predictionsKwh      baseload (house consumption + heat pumps)
     *                            in <b>kWh per interval</b> (not kW).
     *                            Length = planning horizon.
     * @param solarKwh            solar generation forecast in kWh per interval.
     *                            Same length as {@code predictionsKwh}.
     * @param importPrices         grid import prices in EUR/kWh per interval.
     *                            Same length as {@code predictionsKwh}.
     * @param exportPrices         grid export (sell-back) prices in EUR/kWh
     *                            per interval.  Same length.
     * @param predictionTimestamps ISO-8601 string timestamps for each interval.
     *                            Same length.  Included for logging /
     *                            debugging purposes.
     * @param committedLoadKwh    loads that consume grid capacity but are
     *                            <em>not</em> served by the house battery
     *                            (EV charger, Leaf, etc.).  May be {@code null}
     *                            (treat as all-zeros).  Same length.
     * @param allowExport         whether the battery is allowed to export
     *                            energy to the grid.  When {@code false} the
     *                            planner must not set
     *                            {@code dischargeToExportKwh > 0}.
     * @param initialSocPct       current battery state of charge in percent
     *                            (0-100).  {@code null} means use the default
     *                            from {@link BatteryConfig#getInitialSocPct()}.
     * @param context             optional extra data (temperatures, EV
     *                            position, prediction bounds, etc.).  See
     *                            the table in {@link BatteryPlanner} for
     *                            standard keys.  Must ignore unknown keys.
     * @return a plan with exactly one {@link BatteryPlanEntry} per interval
     *         (same length as input arrays), never {@code null}
     */
    List<BatteryPlanEntry> plan(
        double[] predictionsKwh,
        double[] solarKwh,
        double[] importPrices,
        double[] exportPrices,
        List<String> predictionTimestamps,
        double[] committedLoadKwh,
        boolean allowExport,
        Double initialSocPct,
        Map<String, Object> context
    );

    /**
     * Simplified {@link #plan} call with all optional parameters set to
     * defaults: no committed load, export allowed, default initial SoC,
     * no context.
     */
    default List<BatteryPlanEntry> plan(
            double[] predictionsKwh,
            double[] solarKwh,
            double[] importPrices,
            double[] exportPrices,
            List<String> predictionTimestamps) {
        return plan(predictionsKwh, solarKwh, importPrices, exportPrices,
                    predictionTimestamps, null, true, null, null);
    }
}
