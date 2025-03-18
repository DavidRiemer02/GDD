package de.unimarburg.generateddatasetdetector.matching.similarity.label;

import de.unimarburg.generateddatasetdetector.similarity.string.LongestCommonSubsequence;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
public class LongestCommonSubsequenceMatcher extends LabelSimilarityMatcher {
    public LongestCommonSubsequenceMatcher() {
        super(new LongestCommonSubsequence());
    }
}
