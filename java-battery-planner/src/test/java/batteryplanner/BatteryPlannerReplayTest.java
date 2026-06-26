package batteryplanner;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parameterised replay tests that run every registered planner against
 * every JSON fixture in the {@code fixtures/} directory.
 *
 * <h3>How to add a new planner to the test suite</h3>
 * <p>Edit the {@link #discoverPlanners()} method and add a new entry:
 * <pre>
 *   new PlannerEntry("my-planner", new MyBatteryPlanner(TEST_CONFIG))
 * </pre>
 *
 * <p>Then run the tests with Maven:
 * <pre>
 *   cd java-battery-planner
 *   mvn test
 * </pre>
 *
 * <p>The test provides four checks:
 * <ol>
 *   <li>{@link #testNoViolations} — SoC must stay within min/max bounds.</li>
 *   <li>{@link #testFiniteCost} — costs must be finite (no NaN/Inf).</li>
 *   <li>{@link #testNotWorseThanBaseline} — planner must not cost more
 *       than 2× the no-battery baseline (catches degenerate planners).</li>
 *   <li>{@link #testQuick} — a short 96-interval smoke test for quick
 *       iteration during development.</li>
 * </ol>
 */
public class BatteryPlannerReplayTest {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .registerModule(new JavaTimeModule());

    private static final BatteryConfig TEST_CONFIG = new BatteryConfig(
        50.0,   // capacity Kwh
        10.0,   // min SoC %
        90.0,   // max SoC %
        10.0,   // initial SoC %
        10.0,   // max charge kW
        10.0,   // max discharge kW
        0.95,   // charge efficiency
        0.95    // discharge efficiency
    );

    private static List<FixtureEntry> discoverFixtures() {
        Path fixtureDir = Path.of("fixtures");
        if (!Files.isDirectory(fixtureDir)) {
            fixtureDir = Path.of("../fixtures");  // when running from subdir
        }
        if (!Files.isDirectory(fixtureDir)) {
            fixtureDir = Path.of("java-battery-planner/fixtures");  // when running from workspace
        }
        if (!Files.isDirectory(fixtureDir)) {
            System.err.println("WARNING: Fixtures directory not found. Checked fixtures/, ../fixtures/, java-battery-planner/fixtures/");
            return List.of();
        }

        List<FixtureEntry> entries = new ArrayList<>();

        try (Stream<Path> files = Files.list(fixtureDir)) {
            List<Path> jsonFiles = files
                .filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().endsWith(".json"))
                .sorted()
                .collect(Collectors.toList());

            for (Path jsonPath : jsonFiles) {
                try {
                    Map<String, Object> raw = MAPPER.readValue(
                        jsonPath.toFile(),
                        new TypeReference<Map<String, Object>>() {});
                    FixtureData fixture = new FixtureData(raw);
                    String name = jsonPath.getFileName().toString().replace(".json", "");
                    entries.add(new FixtureEntry(name, jsonPath.toString(), fixture));
                } catch (Exception e) {
                    System.err.println("WARNING: Could not load " + jsonPath + ": " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("WARNING: Could not list fixtures: " + e.getMessage());
        }

        return entries;
    }

    /**
     * Register planners here.  Add a new {@link PlannerEntry} for each
     * planner you want to test against every fixture.
     */
    private static List<PlannerEntry> discoverPlanners() {
        return List.of(
            new PlannerEntry("heuristic", new HeuristicBatteryPlanner(TEST_CONFIG))
        );
    }

    private static Stream<Arguments> allCombinations() {
        List<FixtureEntry> fixtures = discoverFixtures();
        List<PlannerEntry> planners = discoverPlanners();

        if (fixtures.isEmpty()) {
            return Stream.of(Arguments.arguments(
                new FixtureEntry("none", "", null),
                new PlannerEntry("none", null)
            ));
        }

        List<Arguments> combinations = new ArrayList<>();
        for (FixtureEntry f : fixtures) {
            for (PlannerEntry p : planners) {
                combinations.add(Arguments.arguments(f, p));
            }
        }
        return combinations.stream();
    }

    @ParameterizedTest(name = "{0} / {1}")
    @MethodSource("allCombinations")
    @DisplayName("No SoC violation replay")
    void testNoViolations(FixtureEntry fixture, PlannerEntry planner) {
        if (fixture.fixture == null || planner.planner == null) return;

        BatteryReplaySimulator sim = new BatteryReplaySimulator(fixture.fixture);

        if (!sim.hasMeasurements()) return;
        if (!sim.hasArchive()) return;

        int horizon = sim.getMeasurements().size();
        ReplayResult result = sim.simulate(
            planner.planner, planner.name, TEST_CONFIG, horizon);

        assertTrue(result.isSuccess(), "Replay failed: " + result.getError());
        assertEquals(0, result.getSocViolations(),
            "SoC violations: " + result.getSocViolationDetails());
        assertTrue(result.getIntervalsRun() > 0, "No intervals were simulated");

        System.out.printf("  %-22s %-5s  savings=%6.1f%%  soc=%5.1f%%  cost=%.3f  base=%.3f  viol=%d%n",
            planner.name, fixture.name,
            result.getSavingsPct(), result.getFinalSocPct(),
            result.getCostWithBatteryEur(), result.getCostNoBatteryEur(),
            result.getSocViolations());
    }

    @ParameterizedTest(name = "{0} / {1}")
    @MethodSource("allCombinations")
    @DisplayName("Finite cost replay")
    void testFiniteCost(FixtureEntry fixture, PlannerEntry planner) {
        if (fixture.fixture == null || planner.planner == null) return;

        BatteryReplaySimulator sim = new BatteryReplaySimulator(fixture.fixture);
        if (!sim.hasMeasurements()) return;
        if (!sim.hasArchive()) return;

        int horizon = sim.getMeasurements().size();
        ReplayResult result = sim.simulate(
            planner.planner, planner.name, TEST_CONFIG, horizon);

        assertTrue(result.isSuccess());
        assertTrue(Double.isFinite(result.getCostWithBatteryEur()),
            "Cost with battery not finite: " + result.getCostWithBatteryEur());
        assertTrue(Double.isFinite(result.getCostNoBatteryEur()),
            "Baseline cost not finite: " + result.getCostNoBatteryEur());

        System.out.printf("  %-22s %-5s  savings=%6.1f%%  cost=%.3f  base=%.3f%n",
            planner.name, fixture.name,
            result.getSavingsPct(), result.getCostWithBatteryEur(), result.getCostNoBatteryEur());
    }

    @ParameterizedTest(name = "{0} / {1}")
    @MethodSource("allCombinations")
    @DisplayName("Cost not worse than 2x baseline")
    void testNotWorseThanBaseline(FixtureEntry fixture, PlannerEntry planner) {
        if (fixture.fixture == null || planner.planner == null) return;

        BatteryReplaySimulator sim = new BatteryReplaySimulator(fixture.fixture);
        if (!sim.hasMeasurements()) return;
        if (!sim.hasArchive()) return;

        int horizon = sim.getMeasurements().size();
        ReplayResult result = sim.simulate(
            planner.planner, planner.name, TEST_CONFIG, horizon);

        assertTrue(result.isSuccess());

        double baseline = result.getCostNoBatteryEur();
        double plannerCost = result.getCostWithBatteryEur();
        double maxAcceptable = baseline * 2.0;

        assertTrue(plannerCost <= maxAcceptable,
            String.format("Planner cost %.2f exceeds 2x baseline (%.2f)", plannerCost, maxAcceptable));
    }

    @ParameterizedTest(name = "{0} / {1}")
    @MethodSource("allCombinations")
    @DisplayName("Quick 96-interval smoke test")
    void testQuick(FixtureEntry fixture, PlannerEntry planner) {
        if (fixture.fixture == null || planner.planner == null) return;

        BatteryReplaySimulator sim = new BatteryReplaySimulator(fixture.fixture);
        if (!sim.hasMeasurements()) return;

        ReplayResult result = sim.simulate(
            planner.planner, planner.name, TEST_CONFIG, 96);

        assertTrue(result.isSuccess(), "Quick replay failed: " + result.getError());
        assertEquals(0, result.getSocViolations(),
            "SoC violations: " + result.getSocViolationDetails());
        assertTrue(Double.isFinite(result.getSavingsPct()), "Savings not finite");
        assertTrue(Double.isFinite(result.getCostWithBatteryEur()), "Cost not finite");

        System.out.printf("  %-22s %-5s  savings=%6.1f%%  cost=%.3f  base=%.3f%n",
            planner.name, fixture.name,
            result.getSavingsPct(), result.getCostWithBatteryEur(), result.getCostNoBatteryEur());
    }

    static class FixtureEntry {
        final String name;
        final String path;
        final FixtureData fixture;
        FixtureEntry(String name, String path, FixtureData fixture) {
            this.name = name;
            this.path = path;
            this.fixture = fixture;
        }
        @Override
        public String toString() { return name; }
    }

    static class PlannerEntry {
        final String name;
        final BatteryPlanner planner;
        PlannerEntry(String name, BatteryPlanner planner) {
            this.name = name;
            this.planner = planner;
        }
        @Override
        public String toString() { return name; }
    }
}
