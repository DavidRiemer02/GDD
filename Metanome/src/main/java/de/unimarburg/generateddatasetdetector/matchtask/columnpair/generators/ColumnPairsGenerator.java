package de.unimarburg.generateddatasetdetector.matchtask.columnpair.generators;

import de.unimarburg.generateddatasetdetector.matchtask.columnpair.ColumnPair;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;

import java.util.List;

public interface ColumnPairsGenerator {
    public List<ColumnPair> generateCandidates(TablePair tablePair);
}
