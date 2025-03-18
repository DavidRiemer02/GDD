package de.unimarburg.generateddatasetdetector.matching.similarity.label;

import de.unimarburg.generateddatasetdetector.similarity.string.Levenshtein;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class LevenshteinMatcher extends LabelSimilarityMatcher {
    public LevenshteinMatcher() {
        super(new Levenshtein());
    }
}
