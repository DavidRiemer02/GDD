package de.unimarburg.generateddatasetdetector.matching.similarity.label;

import de.unimarburg.generateddatasetdetector.matching.TablePairMatcher;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;
import de.unimarburg.generateddatasetdetector.data.Table;
import de.unimarburg.generateddatasetdetector.similarity.SimilarityMeasure;
import lombok.NoArgsConstructor;

@NoArgsConstructor
public abstract class LabelSimilarityMatcher extends TablePairMatcher {
    private SimilarityMeasure<String> similarityMeasure;

    public LabelSimilarityMatcher(SimilarityMeasure<String> similarityMeasure) {
        super();
        this.similarityMeasure = similarityMeasure;
    }

    @Override
    public float[][] match(TablePair tablePair) {
        Table sourceTable = tablePair.getSourceTable();
        Table targetTable = tablePair.getTargetTable();
        float[][] simMatrix = tablePair.getEmptySimMatrix();
        for (int i = 0; i < sourceTable.getNumColumns(); i++) {
            for (int j = 0; j < targetTable.getNumColumns(); j++) {
                simMatrix[i][j] = similarityMeasure.compare(sourceTable.getLabels().get(i), targetTable.getLabels().get(j));
            }
        }
        return simMatrix;
    }
}