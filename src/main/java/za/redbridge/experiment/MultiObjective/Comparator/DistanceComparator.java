package za.redbridge.experiment.MultiObjective.Comparator;

import za.redbridge.experiment.MultiObjective.MultiObjectiveGenome;

import java.util.Comparator;

public class DistanceComparator<T extends MultiObjectiveGenome> implements Comparator<T>
{



    /**
     * Compares one single objective value of two individuals. Which objective
     * is compared is given by an index returned by #getObjectiveIndex. For
     * example, if this index is 2, then the two individuals are compared
     * against the third objective value (indices start by 0). Bigger values
     * are considered better.
     *
     * @param o1 - First individual.
     * @param o2 - Second individual.
     * @return -1 if o1 is better than o2, 1 if o2 is better than o1 and 0 if
     * both individual have the same objective value for the given objective
     * index.
     */
    @Override
    public int compare(MultiObjectiveGenome o1, MultiObjectiveGenome o2)
    {
        double o1Value = o1.getDistance();
        double o2Value = o2.getDistance();

        if (o1Value > o2Value)
        {
            return -1;
        } else if(o1Value < o2Value)
        {
            return 1;
        }
        return 0;
    }



}
