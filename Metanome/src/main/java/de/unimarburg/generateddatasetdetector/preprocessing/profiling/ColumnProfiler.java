package de.unimarburg.generateddatasetdetector.preprocessing.profiling;

import de.unimarburg.generateddatasetdetector.data.Column;

public interface ColumnProfiler {
    public void profile(Column column);
}
