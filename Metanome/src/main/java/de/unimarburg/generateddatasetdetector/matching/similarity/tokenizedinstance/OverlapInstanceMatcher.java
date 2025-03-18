package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedinstance;

import lombok.EqualsAndHashCode;
import de.unimarburg.generateddatasetdetector.similarity.set.Overlap;


@EqualsAndHashCode(callSuper = true)
public class OverlapInstanceMatcher extends TokenizedInstanceSimilarityMatcher {
    public OverlapInstanceMatcher() {
        super(new Overlap<>());
    }
}
