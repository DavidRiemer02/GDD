package de.unimarburg.generateddatasetdetector.matchtask.tablepair.generators;

import de.unimarburg.generateddatasetdetector.data.Scenario;
import de.unimarburg.generateddatasetdetector.matchtask.tablepair.TablePair;

import java.util.List;

public interface TablePairsGenerator {
    public List<TablePair> generateCandidates(Scenario scenario);
}
