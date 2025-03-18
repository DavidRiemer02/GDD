package de.unimarburg.generateddatasetdetector.matching.similarity.label;

import de.unimarburg.generateddatasetdetector.similarity.string.Cosine;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class CosineMatcher extends LabelSimilarityMatcher {
    public CosineMatcher() {
        super(new Cosine());
    }
}
