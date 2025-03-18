package de.unimarburg.generateddatasetdetector.matching.similarity.label;

import de.unimarburg.generateddatasetdetector.similarity.string.Hamming;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class HammingMatcher extends LabelSimilarityMatcher {
    public HammingMatcher() {
        super(new Hamming());
    }
}
