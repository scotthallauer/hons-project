package za.redbridge.experiment.NEAT;

import org.encog.engine.network.activation.*;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.util.obj.ChooseObject;

/**
 * NEATPopulation implementation with a weight range of -1.0 to 1.0.
 *
 * Created by jamie on 2014/10/14.
 */
public class NEATPopulation extends org.encog.neural.neat.NEATPopulation {

    private static final long serialVersionUID = 4907092837215248072L;

    public NEATPopulation() {
    }

    public NEATPopulation(int inputCount, int outputCount, int populationSize) {
        super(inputCount, outputCount, populationSize);
    }

    public NEATPopulation(Substrate theSubstrate, int populationSize) {
        super(theSubstrate, populationSize);
    }



    @Override
    public double getWeightRange() {
        return 5.0;
    }
}
