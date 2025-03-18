package de.unimarburg.generateddatasetdetector.utils;

import de.unimarburg.generateddatasetdetector.Main;
import de.unimarburg.generateddatasetdetector.data.Dataset;
import de.unimarburg.generateddatasetdetector.matchtask.MatchTask;
import de.unimarburg.generateddatasetdetector.matchtask.matchstep.MatchStep;

import java.nio.file.Path;

public class ResultsUtils {
    private final static Configuration config = Configuration.getInstance();

    // Base result paths
    public static Path getBaseResultsPath() {
        Configuration config = Configuration.getInstance();
        return Path.of(config.getResultsDir(), StringUtils.dateToString(Main.START_TIMESTAMP));
    }
    public static Path getBaseResultsPathForDataset(Dataset dataset) {
        return getBaseResultsPath().resolve(dataset.getName());
    }
    public static Path getBaseResultsPathForScenario(MatchTask matchTask) {
        return getBaseResultsPathForDataset(matchTask.getDataset()).resolve(matchTask.getScenario().getName());
    }

    // Output paths
    public static Path getOutputPathForScenario(MatchTask matchTask) {
        return getBaseResultsPathForScenario(matchTask).resolve(config.getOutputDir());
    }
    public static Path getOutputPathForMatchStepInScenario(MatchTask matchTask, MatchStep matchStep) {
        return getOutputPathForScenario(matchTask).resolve(matchStep.toString());
    }

    // Cache paths
    public static Path getCachePathForMatchStepInScenario(MatchTask matchTask, MatchStep matchingStep) {
        return Path.of(config.getCacheDir())
                .resolve(matchTask.getDataset().getName())
                .resolve(matchTask.getScenario().getName())
                .resolve(matchingStep.toString());
    }

    // Performance paths
    public static Path getPerformancePathForOverall() {
        return getBaseResultsPath().resolve(config.getPerformanceDir());
    }
    public static Path getPerformancePathForDataset(Dataset dataset) {
        return getBaseResultsPathForDataset(dataset).resolve(config.getPerformanceDir());
    }
    public static Path getPerformancePathForScenario(MatchTask matchTask) {
        return getBaseResultsPathForScenario(matchTask).resolve(config.getPerformanceDir());
    }
}
