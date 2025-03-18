package de.unimarburg.generateddatasetdetector.matching.similarity.instance;

import de.unimarburg.generateddatasetdetector.matching.TablePairMatcher;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;
import de.unimarburg.generateddatasetdetector.matching.Matcher;
import org.apache.commons.lang3.NotImplementedException;

public abstract class InstanceSimilarityMatcher extends TablePairMatcher {
    @Override
    public float[][] match(TablePair tablePair) {
        throw new NotImplementedException("No instance matchers yet, use TokenizedInstanceSimilarityMatcher instead.");
    }
}
