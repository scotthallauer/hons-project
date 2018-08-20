package za.redbridge.experiment.MultiObjective;

import org.encog.ml.CalculateScore;
import org.encog.ml.ea.opp.CompoundOperator;
import org.encog.ml.ea.opp.OperationList;
import org.encog.ml.ea.opp.selection.TruncationSelection;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.opp.NEATMutateAddLink;
import org.encog.neural.neat.training.opp.NEATMutateWeights;
import org.encog.neural.neat.training.opp.links.MutatePerturbLinkWeight;
import org.encog.neural.neat.training.opp.links.MutateResetLinkWeight;
import org.encog.neural.neat.training.opp.links.SelectFixed;
import org.encog.neural.neat.training.opp.links.SelectProportion;
import org.encog.neural.neat.training.species.OriginalNEATSpeciation;
import za.redbridge.experiment.NEATM.NEATMCODEC;
import za.redbridge.experiment.NEATM.NEATMPopulation;
import za.redbridge.experiment.NEATM.sensor.SensorType;
import za.redbridge.experiment.NEATM.training.opp.*;
import za.redbridge.experiment.NEATM.training.opp.sensors.MutatePerturbSensorParameter;
import za.redbridge.experiment.NEATM.training.opp.sensors.SelectSensorsFixed;
import za.redbridge.experiment.NEATM.training.opp.sensors.SelectSensorsType;
import za.redbridge.experiment.NEATM.training.species.NEATMSpeciation;

import static za.redbridge.experiment.NEATM.sensor.SensorType.*;
import static za.redbridge.experiment.NEATM.sensor.SensorType.LOW_RES_CAM;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.ParameterType.*;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.ParameterType.FIELD_OF_VIEW;

public class MultiObjectiveNEATMUtil
{
    public static MultiObjectiveEA constructNEATTrainer(CalculateScore calculateScore, int outputCount,
                                               int populationSize, int numGenerations) {
        final NEATMPopulation pop = new NEATMPopulation(outputCount, populationSize, true);
        pop.reset();
        return constructNEATTrainer(pop, calculateScore, numGenerations);
    }

    /**
     * Construct a NEAT trainer.
     * @param population The population.
     * @param calculateScore The score function.
     * @return The NEAT EA trainer.
     */
    public static MultiObjectiveEA constructNEATTrainer(NEATPopulation population,
                                               CalculateScore calculateScore, int numGenerations) {
        final MultiObjectiveEA result = new MultiObjectiveEA(population, calculateScore);
        NEATMSpeciation speciation = new NEATMSpeciation();
        speciation.setNumGensAllowedNoImprovement(numGenerations);
        speciation.setMaxNumberOfSpecies(population.size());
        result.setSpeciation(speciation);

        // Selection
        result.setSelection(new TruncationSelection(result, 0.3));

        // Create compound operator for weight mutation
        CompoundOperator weightMutation = new CompoundOperator();
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectFixed(1), new MutatePerturbLinkWeight(0.004)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectFixed(2), new MutatePerturbLinkWeight(0.004)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectFixed(3), new MutatePerturbLinkWeight(0.004)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectProportion(0.004),
                        new MutatePerturbLinkWeight(0.004)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectFixed(1), new MutatePerturbLinkWeight(0.2)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectFixed(2), new MutatePerturbLinkWeight(0.2)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectFixed(3), new MutatePerturbLinkWeight(0.2)));
        weightMutation.getComponents().add(0.1125,
                new NEATMutateWeights(new SelectProportion(0.02),
                        new MutatePerturbLinkWeight(0.2)));
        weightMutation.getComponents().add(0.03,
                new NEATMutateWeights(new SelectFixed(1), new MutateResetLinkWeight()));
        weightMutation.getComponents().add(0.03,
                new NEATMutateWeights(new SelectFixed(2), new MutateResetLinkWeight()));
        weightMutation.getComponents().add(0.03,
                new NEATMutateWeights(new SelectFixed(3), new MutateResetLinkWeight()));
        weightMutation.getComponents().add(0.01,
                new NEATMutateWeights(new SelectProportion(0.02), new MutateResetLinkWeight()));
        weightMutation.getComponents().finalizeStructure();

        // Champ operator limits mutation operator on best genome (seems unused for now)
        result.setChampMutation(weightMutation);

        // Add all the operators, probability should sum to 1
        result.addOperation(0.32, new NEATMCrossover());
        result.addOperation(0.335, weightMutation);
        result.addOperation(0.05, new NEATMMutateAddNode());
        result.addOperation(0.05, new NEATMutateAddLink());
        result.addOperation(0.005, new NEATMMutateRemoveLink());

        // Add the sensor position mutators
        CompoundOperator positionMutation = new CompoundOperator();
        OperationList positionMutationComponents = positionMutation.getComponents();
        //MUTATE BEARING
        //mutate one sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(1), new MutatePerturbSensorParameter(5.0f, BEARING)));
        //mutate two sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(2), new MutatePerturbSensorParameter(5.0f, BEARING)));
        //mutate three sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(3), new MutatePerturbSensorParameter(5.0f, BEARING)));
        //mutate four sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(4), new MutatePerturbSensorParameter(5.0f, BEARING)));
        //MUTATE ORIENTATION
        //mutate one sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(1), new MutatePerturbSensorParameter(5.0f, ORIENTATION)));
        //mutate two sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(2), new MutatePerturbSensorParameter(5.0f, ORIENTATION)));
        //mutate three sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(3), new MutatePerturbSensorParameter(5.0f, ORIENTATION)));
        //mutate four sensors
        positionMutationComponents.add(0.125, new NEATMMutateIndividualSensor(
                new SelectSensorsFixed(4), new MutatePerturbSensorParameter(5.0f, ORIENTATION)));
        positionMutationComponents.finalizeStructure();

        result.addOperation(0.1, positionMutation);

        // Add the sensor field mutators
        CompoundOperator fieldMutation = new CompoundOperator();
        OperationList fieldMutationComponents = fieldMutation.getComponents();
        //RANGE
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(ULTRASONIC), new MutatePerturbSensorParameter(5.0f, RANGE)));
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(PROXIMITY), new MutatePerturbSensorParameter(5.0f, RANGE)));
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(COLOUR_PROXIMITY), new MutatePerturbSensorParameter(5.0f, RANGE)));
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(LOW_RES_CAM), new MutatePerturbSensorParameter(5.0f, RANGE)));
        //FOV
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(ULTRASONIC),
                new MutatePerturbSensorParameter(5.0f, FIELD_OF_VIEW)));
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(PROXIMITY),
                new MutatePerturbSensorParameter(5.0f, FIELD_OF_VIEW)));
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(COLOUR_PROXIMITY),
                new MutatePerturbSensorParameter(5.0f, FIELD_OF_VIEW)));
        fieldMutationComponents.add(0.25, new NEATMMutateSensorGroup(
                new SelectSensorsType(LOW_RES_CAM),
                new MutatePerturbSensorParameter(5.0f, FIELD_OF_VIEW)));

        fieldMutationComponents.finalizeStructure();

        result.addOperation(0.07, fieldMutation);

        // Add sensor mutation
        double connectionDensity = 0.1;
        CompoundOperator addSensorMutation = new CompoundOperator();
        OperationList addSensorComponents = addSensorMutation.getComponents();
        addSensorComponents.add(0.25, new NEATMMutateAddSensor(SensorType.PROXIMITY,
                connectionDensity, new MutateResetLinkWeight()));
        addSensorComponents.add(0.25, new NEATMMutateAddSensor(SensorType.ULTRASONIC,
                connectionDensity, new MutatePerturbLinkWeight(0.2)));
        addSensorComponents.finalizeStructure();
        addSensorComponents.add(0.25, new NEATMMutateAddSensor(SensorType.COLOUR_PROXIMITY,
                connectionDensity, new MutateResetLinkWeight()));
        addSensorComponents.add(0.25, new NEATMMutateAddSensor(SensorType.LOW_RES_CAM,
                connectionDensity, new MutatePerturbLinkWeight(0.2)));
        addSensorComponents.finalizeStructure();

        result.addOperation(0.07, addSensorMutation);


        result.getOperators().finalizeStructure();

        result.setCODEC(new NEATMCODEC());

        return result;
    }

}
