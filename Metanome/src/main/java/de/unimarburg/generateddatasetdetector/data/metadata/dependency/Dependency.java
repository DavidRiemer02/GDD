package de.unimarburg.generateddatasetdetector.data.metadata.dependency;

import de.unimarburg.generateddatasetdetector.data.Column;

import java.util.Collection;

public interface Dependency {
    default void addColumns(StringBuilder sb, Collection<Column> referenced) {
        sb.append("[");
        for (Column column : referenced) {
            sb.append(column.getTable().getName());
            sb.append(".csv.");
            sb.append(column.getLabel());
            sb.append(", ");
        }
        if(!referenced.isEmpty())
            sb.delete(sb.length() - 2, sb.length());
        sb.append("]");
    }
}
