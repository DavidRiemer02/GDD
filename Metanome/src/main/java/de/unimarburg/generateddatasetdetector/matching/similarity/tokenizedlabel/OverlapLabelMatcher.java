package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedlabel;

import lombok.EqualsAndHashCode;
import de.unimarburg.generateddatasetdetector.similarity.set.Overlap;



@EqualsAndHashCode(callSuper = true)
public class OverlapLabelMatcher extends TokenizedLabelSimilarityMatcher {
    public OverlapLabelMatcher() {
        super(new Overlap<>());
    }
}
