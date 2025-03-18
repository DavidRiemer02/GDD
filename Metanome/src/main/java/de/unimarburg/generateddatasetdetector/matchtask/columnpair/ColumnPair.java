package de.unimarburg.generateddatasetdetector.matchtask.columnpair;

import de.unimarburg.generateddatasetdetector.data.Column;
import de.unimarburg.generateddatasetdetector.utils.Configuration;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ColumnPair {
    private final Column sourceColumn;
    private final Column targetColumn;

    public String toString() {
        return sourceColumn.getLabel() + Configuration.getInstance().getDefaultTablePairSeparator() + targetColumn.getLabel();
    }
}
