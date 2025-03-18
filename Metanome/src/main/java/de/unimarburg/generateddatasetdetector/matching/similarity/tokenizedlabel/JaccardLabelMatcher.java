package de.unimarburg.generateddatasetdetector.matching.similarity.tokenizedlabel;

import de.unimarburg.generateddatasetdetector.similarity.set.Jaccard;
import lombok.EqualsAndHashCode;


@EqualsAndHashCode(callSuper = true)
public class JaccardLabelMatcher extends TokenizedLabelSimilarityMatcher {
    public JaccardLabelMatcher() {
        super(new Jaccard<String>());
    }
}
