package de.unimarburg.generateddatasetdetector.boosting;

import de.unimarburg.generateddatasetdetector.matchtask.MatchTask;
import de.unimarburg.generateddatasetdetector.matchtask.matchstep.SimMatrixBoostingStep;

public interface SimMatrixBoosting {
    /**
     * @param matchTask MatchTask to boost sim matrix for
     * @param matchStep Current MatchStep (SimMatrixBoostingStep)
     * @param simMatrix similarity matrix to improve
     * @return An updated (hopefully better) similarity matrix
     */
    public float[][] run(MatchTask matchTask, SimMatrixBoostingStep matchStep, float[][] simMatrix);
}
