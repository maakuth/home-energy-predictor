package batteryplanner;

/**
 * Battery physical parameters and operating constraints.
 *
 * <p>These values come from the Home Assistant battery configuration and
 * the {@code .env} file (transfer fees, min/max SoC, efficiency, etc.).
 * A planner should use {@link #getMinSocPct()} and {@link #getMaxSocPct()}
 * as hard operating bounds and never schedule charge/discharge that
 * would violate them.
 *
 * <p>The default constructor provides sensible defaults (50 kWh, 10-90 %,
 * 10 kW, 95 % efficiency) for development and testing.
 */
public class BatteryConfig {
    private final double capacityKwh;
    private final double minSocPct;
    private final double maxSocPct;
    private final double initialSocPct;
    private final double maxChargeKw;
    private final double maxDischargeKw;
    private final double chargeEfficiency;
    private final double dischargeEfficiency;

    public BatteryConfig() {
        this(50.0, 10.0, 90.0, 10.0, 10.0, 10.0, 0.95, 0.95);
    }

    /**
     * @param capacityKwh       usable battery capacity (kWh)
     * @param minSocPct         minimum allowed SoC (%), e.g. 10
     * @param maxSocPct         maximum allowed SoC (%), e.g. 90
     * @param initialSocPct     default starting SoC (%) when the replay begins
     * @param maxChargeKw       max charge power (kW)
     * @param maxDischargeKw    max discharge power (kW)
     * @param chargeEfficiency  round-trip charge leg efficiency (0-1), e.g. 0.95
     * @param dischargeEfficiency round-trip discharge leg efficiency (0-1)
     */
    public BatteryConfig(double capacityKwh, double minSocPct, double maxSocPct,
                         double initialSocPct, double maxChargeKw, double maxDischargeKw,
                         double chargeEfficiency, double dischargeEfficiency) {
        this.capacityKwh = capacityKwh;
        this.minSocPct = minSocPct;
        this.maxSocPct = maxSocPct;
        this.initialSocPct = initialSocPct;
        this.maxChargeKw = maxChargeKw;
        this.maxDischargeKw = maxDischargeKw;
        this.chargeEfficiency = chargeEfficiency;
        this.dischargeEfficiency = dischargeEfficiency;
    }

    public double getCapacityKwh() { return capacityKwh; }
    public double getMinSocPct() { return minSocPct; }
    public double getMaxSocPct() { return maxSocPct; }
    public double getInitialSocPct() { return initialSocPct; }
    public double getMaxChargeKw() { return maxChargeKw; }
    public double getMaxDischargeKw() { return maxDischargeKw; }
    public double getChargeEfficiency() { return chargeEfficiency; }
    public double getDischargeEfficiency() { return dischargeEfficiency; }

    /** Convenience: min SoC expressed in kWh. */
    public double getMinSocKwh() { return capacityKwh * minSocPct / 100.0; }

    /** Convenience: max SoC expressed in kWh. */
    public double getMaxSocKwh() { return capacityKwh * maxSocPct / 100.0; }
}
