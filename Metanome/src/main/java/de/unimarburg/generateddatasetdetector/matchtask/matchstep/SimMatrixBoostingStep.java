package de.unimarburg.generateddatasetdetector.matchtask.matchstep;

import de.unimarburg.generateddatasetdetector.boosting.SimMatrixBoosting;
import de.unimarburg.generateddatasetdetector.evaluation.metric.Metric;
import de.unimarburg.generateddatasetdetector.evaluation.performance.Performance;
import de.unimarburg.generateddatasetdetector.matching.Matcher;
import de.unimarburg.generateddatasetdetector.matchtask.MatchTask;
import de.unimarburg.generateddatasetdetector.utils.Configuration;
import de.unimarburg.generateddatasetdetector.utils.InputReader;
import de.unimarburg.generateddatasetdetector.utils.OutputWriter;
import de.unimarburg.generateddatasetdetector.utils.ResultsUtils;
import lombok.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

@Getter
@EqualsAndHashCode(callSuper = true)
public class SimMatrixBoostingStep extends MatchStep {
    final static Logger log = LogManager.getLogger(SimMatrixBoostingStep.class);

    private final int line;
    private final SimMatrixBoosting simMatrixBoosting;

    public SimMatrixBoostingStep(boolean doSave, boolean doEvaluate, int line, SimMatrixBoosting simMatrixBoosting) {
        super(doSave, doEvaluate);
        this.line = line;
        this.simMatrixBoosting = simMatrixBoosting;
    }

    @Override
    public String toString() {
        return super.toString() + "Line" + line;
    }

    @Override
    public void run(MatchTask matchTask) {
        log.debug("Running similarity matrix boosting (line=" + this.line + ") for scenario: " + matchTask.getScenario().getPath());
        List<Matcher> matchers = matchTask.getMatchersForLine(this.line);
        for (Matcher matcher : matchers) {
            float[][] boostedSimMatrix = null;
            if ((line == 1 && Configuration.getInstance().isReadCacheSimMatrixBoostingOnFirstLineMatchers()) ||
                    line == 2 && Configuration.getInstance().isReadCacheSimMatrixBoostingOnSecondLineMatchers()) {
                boostedSimMatrix = InputReader.readCache(matchTask,this, matcher);
            }
            if (boostedSimMatrix == null) {
                log.debug("Processing " + this.line + ". line sim matrix boosting on matcher: " + matcher.toString());
                boostedSimMatrix = this.simMatrixBoosting.run(matchTask, this, matchTask.getSimMatrixFromPreviousMatchStep(this, matcher));
            }
            matchTask.setSimMatrix(this, matcher, boostedSimMatrix);
        }
    }

    @Override
    public void save(MatchTask matchTask) {
        // write cache
        if ((line == 1 && Configuration.getInstance().isWriteCacheSimMatrixBoostingOnFirstLineMatchers()) ||
                line == 2 && Configuration.getInstance().isWriteCacheSimMatrixBoostingOnSecondLineMatchers()) {
            log.debug("Caching similarity matrix boosting (line=" + this.line + ") output for scenario: " + matchTask.getScenario().getPath());

            Path outputMatchStepPath = ResultsUtils.getCachePathForMatchStepInScenario(matchTask, this);

            for (Matcher matcher : matchTask.getMatchersForLine(this.line)) {
                float[][] simMatrix = matchTask.getSimMatrix(this, matcher);
                OutputWriter.writeSimMatrix(outputMatchStepPath, matchTask, matcher.toString(), simMatrix);
                matchTask.incrementCacheWrite();
            }
        }

        // write results
        if ((line == 1 && Configuration.getInstance().isSaveOutputSimMatrixBoostingOnFirstLineMatchers()) ||
                line == 2 && Configuration.getInstance().isSaveOutputSimMatrixBoostingOnSecondLineMatchers()) {
            log.debug("Saving similarity matrix boosting (line=" + this.line + ") output for scenario: " + matchTask.getScenario().getPath());

            Path outputMatchStepPath = ResultsUtils.getOutputPathForMatchStepInScenario(matchTask, this);

            for (Matcher matcher : matchTask.getMatchersForLine(this.line)) {
                float[][] simMatrix = matchTask.getSimMatrix(this, matcher);
                OutputWriter.writeSimMatrix(outputMatchStepPath, matchTask, matcher.toString(), simMatrix);
            }
        }
    }

    @Override
    public void evaluate(MatchTask matchTask) {
        if ((line == 1 && !Configuration.getInstance().isEvaluateSimMatrixBoostingOnFirstLineMatchers()) ||
                line == 2 && !Configuration.getInstance().isEvaluateSimMatrixBoostingOnSecondLineMatchers()) {
            return;
        }
        log.debug("Evaluating similarity matrix boosting (line=" + this.line + ") output for scenario: " + matchTask.getScenario().getPath());


        for (Matcher matcher : matchTask.getMatchersForLine(this.line)) {
            Map<Metric, Performance> performances = matchTask.getEvaluator().evaluate(matchTask.getSimMatrix(this, matcher));
            for (Metric metric : performances.keySet()) {
                matchTask.setPerformanceForMatcher(metric, this, matcher, performances.get(metric));
            }
        }
    }
}
