package de.unimarburg.generateddatasetdetector.matching;

import de.unimarburg.generateddatasetdetector.data.Database;
import de.unimarburg.generateddatasetdetector.matchtask.MatchTask;
import de.unimarburg.generateddatasetdetector.matchtask.matchstep.MatchStep;
import de.unimarburg.generateddatasetdetector.matchtask.matchstep.MatchingStep;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;
import de.unimarburg.generateddatasetdetector.utils.ArrayUtils;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

import java.util.List;

@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
public abstract class TablePairMatcher extends Matcher {
    public abstract float[][] match(TablePair tablePair);

    @Override
    public float[][] match(MatchTask matchTask, MatchingStep matchStep) {
        List<TablePair> tablePairs = matchTask.getTablePairs();

        float[][] simMatrix = matchTask.getEmptySimMatrix();

        for (TablePair tablePair : tablePairs) {
            float[][] tablePairSimMatrix = this.match(tablePair);
            int sourceTableOffset = tablePair.getSourceTable().getOffset();
            int targetTableOffset = tablePair.getTargetTable().getOffset();
            ArrayUtils.insertSubmatrixInMatrix(tablePairSimMatrix, simMatrix, sourceTableOffset, targetTableOffset);
        }

        return simMatrix;
    }
}
