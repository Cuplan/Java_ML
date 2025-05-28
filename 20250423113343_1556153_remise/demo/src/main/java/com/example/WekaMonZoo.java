// WekaMonZoo.java (Entraîne et sauvegarde le modèle)
package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.SerializationHelper;

public class WekaMonZoo {
    public static void main(String[] args) throws Exception {
        // Ici on déclare un datasource (notre arff) et on l'instancie (lecture par le programme)
        DataSource source = new DataSource("src/main/ressources/zoo.arff");
        Instances data = source.getDataSet();

        // Supprimer l'attribut 'animal' s'il est de type STRING. ON CLEAN!!! 
        if (data.attribute("animal") != null && data.attribute("animal").isString()) {
            data.deleteAttributeAt(data.attribute("animal").index());
        }
        // on spécifie quelle attribut a prédire correctement ! (La catégorie de l'animal)
        data.setClassIndex(data.numAttributes() - 1);

        // On invoque notre algo RandomForest (importée) par défaut; c'est un arbe de pensé 
        Classifier cls = new RandomForest();
        // TRAINING! 
        cls.buildClassifier(data);

        // À L'aide d'une autre library, on save notre modle en .cls 
        SerializationHelper.write("zoo-model.model", cls);
        System.out.println("Modèle entraîné et sauvegardé.");
    }
}
