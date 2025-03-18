package de.unimarburg.generateddatasetdetector.data.metadata.dependency;

import de.unimarburg.generateddatasetdetector.data.Column;
import de.unimarburg.generateddatasetdetector.data.metadata.PdepTuple;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;



import java.util.Collection;

@Data
public class FunctionalDependency implements Dependency{
    Collection<Column> determinant;
    Column dependant;
    PdepTuple pdepTuple;

    public FunctionalDependency(Collection<Column> left, Column right){
        this.determinant = left;
        this.dependant = right;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        addColumns(sb, determinant);
        sb.append(" --> ");
        sb.append(dependant.getTable().getName());
        sb.append(".csv.");
        sb.append(dependant.getLabel());
        return sb.toString();
    }

    //Get max length of dependency
    public int getLength(){
        return determinant.size() + 1;
    }
}
