package batteryplanner;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One interval of a battery dispatch plan produced by {@link BatteryPlanner}.
 *
 * <p>The entry records what the battery did during this 15-minute interval,
 * the resulting state of charge, and the net grid exchange.
 *
 * <p>All energy values are in <b>kWh</b> (not kW).  To convert a kW power
 * rate to kWh over the 15-minute interval, multiply by 0.25.
 *
 * <p>Energy conservation relationship (approximately):
 * <pre>
 *   chargeFromSolarKwh + chargeFromGridKwh  ≈  dischargeToLoadKwh + dischargeToExportKwh + ΔSoC
 * </pre>
 * where ΔSoC is the change in {@code socKwh} over the interval (with
 * efficiency losses folded into the charge/discharge branches).
 *
 * @see BatteryPlanner
 */
public class BatteryPlanEntry {
    private String timestamp;
    private String batteryAction;
    private double batteryPowerKw;
    private double chargeFromSolarKwh;
    private double chargeFromGridKwh;
    private double dischargeToLoadKwh;
    private double dischargeToExportKwh;
    private double socKwh;
    private double socPct;
    private double gridImportKwh;
    private double gridExportKwh;
    private double estimatedHourCost;
    private double estimatedHourSavings;
    private double netLoadWithoutBatteryKwh;

    public BatteryPlanEntry() {}

    public BatteryPlanEntry(String timestamp, String batteryAction, double batteryPowerKw,
                            double chargeFromSolarKwh, double chargeFromGridKwh,
                            double dischargeToLoadKwh, double dischargeToExportKwh,
                            double socKwh, double socPct, double gridImportKwh,
                            double gridExportKwh, double estimatedHourCost,
                            double estimatedHourSavings, double netLoadWithoutBatteryKwh) {
        this.timestamp = timestamp;
        this.batteryAction = batteryAction;
        this.batteryPowerKw = batteryPowerKw;
        this.chargeFromSolarKwh = chargeFromSolarKwh;
        this.chargeFromGridKwh = chargeFromGridKwh;
        this.dischargeToLoadKwh = dischargeToLoadKwh;
        this.dischargeToExportKwh = dischargeToExportKwh;
        this.socKwh = socKwh;
        this.socPct = socPct;
        this.gridImportKwh = gridImportKwh;
        this.gridExportKwh = gridExportKwh;
        this.estimatedHourCost = estimatedHourCost;
        this.estimatedHourSavings = estimatedHourSavings;
        this.netLoadWithoutBatteryKwh = netLoadWithoutBatteryKwh;
    }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String v) { this.timestamp = v; }

    /** One of: idle, charge_solar, charge_grid, charge_mixed, discharge_load, discharge_export, discharge_mixed. */
    public String getBatteryAction() { return batteryAction; }
    public void setBatteryAction(String v) { this.batteryAction = v; }

    /** Net battery power (positive = charging, negative = discharging) in kW. */
    public double getBatteryPowerKw() { return batteryPowerKw; }
    public void setBatteryPowerKw(double v) { this.batteryPowerKw = v; }

    /** Energy drawn from surplus solar to charge battery, kWh. */
    public double getChargeFromSolarKwh() { return chargeFromSolarKwh; }
    public void setChargeFromSolarKwh(double v) { this.chargeFromSolarKwh = v; }

    /** Energy drawn from the grid to charge battery, kWh. */
    public double getChargeFromGridKwh() { return chargeFromGridKwh; }
    public void setChargeFromGridKwh(double v) { this.chargeFromGridKwh = v; }

    /** Energy discharged from battery to serve house load, kWh. */
    public double getDischargeToLoadKwh() { return dischargeToLoadKwh; }
    public void setDischargeToLoadKwh(double v) { this.dischargeToLoadKwh = v; }

    /** Energy discharged from battery and exported to grid, kWh. */
    public double getDischargeToExportKwh() { return dischargeToExportKwh; }
    public void setDischargeToExportKwh(double v) { this.dischargeToExportKwh = v; }

    /** Battery state of charge at the end of the interval in kWh. */
    public double getSocKwh() { return socKwh; }
    public void setSocKwh(double v) { this.socKwh = v; }

    /** Battery state of charge at the end of the interval in percent (0-100). */
    public double getSocPct() { return socPct; }
    public void setSocPct(double v) { this.socPct = v; }

    /** Net energy imported from the grid in this interval, kWh. */
    public double getGridImportKwh() { return gridImportKwh; }
    public void setGridImportKwh(double v) { this.gridImportKwh = v; }

    /** Net energy exported to the grid in this interval, kWh. */
    public double getGridExportKwh() { return gridExportKwh; }
    public void setGridExportKwh(double v) { this.gridExportKwh = v; }

    /** Projected cost of this interval's grid exchange with battery, EUR. */
    public double getEstimatedHourCost() { return estimatedHourCost; }
    public void setEstimatedHourCost(double v) { this.estimatedHourCost = v; }

    /** Savings achieved in this interval relative to no-battery baseline, EUR. */
    public double getEstimatedHourSavings() { return estimatedHourSavings; }
    public void setEstimatedHourSavings(double v) { this.estimatedHourSavings = v; }

    /** The net load this interval <em>before</em> battery action (load − solar), kWh. */
    public double getNetLoadWithoutBatteryKwh() { return netLoadWithoutBatteryKwh; }
    public void setNetLoadWithoutBatteryKwh(double v) { this.netLoadWithoutBatteryKwh = v; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("timestamp", timestamp);
        m.put("batteryAction", batteryAction);
        m.put("batteryPowerKw", batteryPowerKw);
        m.put("chargeFromSolarKwh", chargeFromSolarKwh);
        m.put("chargeFromGridKwh", chargeFromGridKwh);
        m.put("dischargeToLoadKwh", dischargeToLoadKwh);
        m.put("dischargeToExportKwh", dischargeToExportKwh);
        m.put("socKwh", socKwh);
        m.put("socPct", socPct);
        m.put("gridImportKwh", gridImportKwh);
        m.put("gridExportKwh", gridExportKwh);
        m.put("estimatedHourCost", estimatedHourCost);
        m.put("estimatedHourSavings", estimatedHourSavings);
        m.put("netLoadWithoutBatteryKwh", netLoadWithoutBatteryKwh);
        return m;
    }

    @Override
    public String toString() {
        return String.format("BatteryPlanEntry[%s action=%s soc=%.1f%% gridIn=%.3f gridOut=%.3f cost=%.3f]",
                timestamp, batteryAction, socPct, gridImportKwh, gridExportKwh, estimatedHourCost);
    }
}
