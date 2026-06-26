package batteryplanner;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The result of a single replay simulation run.
 *
 * <p>Contains financial metrics (cost with battery, baseline cost without
 * battery, savings), any SoC limit violations, and the final SoC.
 */
public class ReplayResult {
    private boolean success;
    private String error;
    private String plannerType;
    private int intervalsRun;
    private int socViolations;
    private List<Map<String, Object>> socViolationDetails;
    private double costWithBatteryEur;
    private double costNoBatteryEur;
    private double savingsEur;
    private double savingsPct;
    private double finalSocPct;

    public ReplayResult() {
        this.socViolationDetails = new ArrayList<>();
    }

    public boolean isSuccess() { return success; }
    public void setSuccess(boolean v) { this.success = v; }

    public String getError() { return error; }
    public void setError(String v) { this.error = v; }

    public String getPlannerType() { return plannerType; }
    public void setPlannerType(String v) { this.plannerType = v; }

    public int getIntervalsRun() { return intervalsRun; }
    public void setIntervalsRun(int v) { this.intervalsRun = v; }

    public int getSocViolations() { return socViolations; }
    public void setSocViolations(int v) { this.socViolations = v; }

    public List<Map<String, Object>> getSocViolationDetails() { return socViolationDetails; }
    public void setSocViolationDetails(List<Map<String, Object>> v) { this.socViolationDetails = v; }

    public double getCostWithBatteryEur() { return costWithBatteryEur; }
    public void setCostWithBatteryEur(double v) { this.costWithBatteryEur = v; }

    public double getCostNoBatteryEur() { return costNoBatteryEur; }
    public void setCostNoBatteryEur(double v) { this.costNoBatteryEur = v; }

    public double getSavingsEur() { return savingsEur; }
    public void setSavingsEur(double v) { this.savingsEur = v; }

    public double getSavingsPct() { return savingsPct; }
    public void setSavingsPct(double v) { this.savingsPct = v; }

    public double getFinalSocPct() { return finalSocPct; }
    public void setFinalSocPct(double v) { this.finalSocPct = v; }

    @Override
    public String toString() {
        return String.format("ReplayResult[success=%s planner=%s intervals=%d violations=%d " +
                "cost=%.3f base=%.3f savings=%.1f%% finalSoc=%.1f%%]",
                success, plannerType, intervalsRun, socViolations,
                costWithBatteryEur, costNoBatteryEur, savingsPct, finalSocPct);
    }
}
