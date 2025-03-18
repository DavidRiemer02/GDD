package de.unimarburg.generateddatasetdetector.matching.sota;

import de.unimarburg.generateddatasetdetector.matchtask.MatchTask;
import de.unimarburg.generateddatasetdetector.matchtask.matchstep.MatchStep;
import de.unimarburg.generateddatasetdetector.matchtask.matchstep.MatchingStep;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;
import de.unimarburg.generateddatasetdetector.matching.Matcher;
import lombok.*;

import java.util.Random;

/**
 * Dummy implementation for a state-of-the-art matcher which outputs a randomized similarity matrix.
 */
@NoArgsConstructor
@AllArgsConstructor
public class RandomMatcher extends Matcher {
    @Getter
    @Setter
    private long seed;

    @Override
    public float[][] match(MatchTask matchTask, MatchingStep matchStep) {
        float[][] simMatrix = matchTask.getEmptySimMatrix();
        Random random = new Random(this.seed);
        for (int i = 0; i < matchTask.getNumSourceColumns(); i++) {
            for (int j = 0; j < matchTask.getNumTargetColumns(); j++) {
                simMatrix[i][j] = random.nextFloat();
            }
        }
        return simMatrix;
    }
}
