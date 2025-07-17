// WekaStats.java (Évalue le modèle sur des données inédites)
package com.example;

import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

public class WekaStats {
    public static void main(String[] args) throws Exception {
        // Initialisation d'un nouveau Datasource avec un arff de test (différent de
        // celui d'entrainement!)
        DataSource source = new DataSource("src/main/ressources/zoo_test.arff");
        Instances testData = source.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        // si les datas sont bieng
        if (testData.attribute("animal") != null && testData.attribute("animal").isString()) {
            testData.deleteAttributeAt(testData.attribute("animal").index());

            // On call notre modele
            Classifier cls = (Classifier) SerializationHelper.read("zoo-model.model");

            // On initialise notre modele pour être évalué
            Evaluation eval = new Evaluation(testData);
            // EXAMEN EN COURS !!!
            eval.evaluateModel(cls, testData);

            // Résultats !
            System.out.println(eval.toSummaryString("=== Résumé ===", false));
            System.out.println(eval.toClassDetailsString("=== Détails par classe ==="));
            System.out.println(eval.toMatrixString("=== Matrice de confusion ==="));
        }

        /* Résultats attendus 
         * === Résumé ===
         * Sur 23 animaux, 21 sont bient classifié. 
         * Correctly Classified Instances 21 91.3043 % 
         * Incorrectly Classified Instances 2 8.6957 %
         * 
         * Kappa (0,8976) : mesure la qualité de la classification en tenant compte du hasard ; proche de 1, c’est très bon.
         * Kappa statistic 0.8976
         * 
         * MAE (0,0757) et RMSE (0,1497) : erreurs moyennes de prédiction ; plus c’est bas, mieux c’est.
         * Mean absolute error 0.0757
         * Root mean squared error 0.1497
         * result en % 
         * Relative absolute error 31.1775 %
         * Root relative squared error 43.0232 %
         * 
         * Le nombre de data dans le dataset (petite base)
         * Total Number of Instances 23
         * 
         * === Détails par classe ===
         * TP Rate FP Rate Precision Recall F-Measure MCC ROC Area PRC Area Class
         * 1,000 0,000 1,000 1,000 1,000 1,000 1,000 1,000 mammal
         * 1,000 0,000 1,000 1,000 1,000 1,000 1,000 1,000 bird
         * 0,667 0,000 1,000 0,667 0,800 0,797 1,000 1,000 reptile
         * 1,000 0,000 1,000 1,000 1,000 1,000 1,000 1,000 fish
         * 1,000 0,048 0,667 1,000 0,800 0,797 1,000 1,000 amphibian
         * 0,667 0,000 1,000 0,667 0,800 0,797 1,000 1,000 insect
         * 1,000 0,050 0,750 1,000 0,857 0,844 0,983 0,917 invertebrate
         * Weighted Avg. 0,913 0,011 0,938 0,913 0,912 0,909 0,998 0,989
         * 
         * 
         *  TP rate = 1.00 = 100% correctement prédit, on peut voir que le modele a du mal a différencier 
         * les reptiles des insectes ! 2 sont biens classés sur 3  Pareil pour amphibiens et invertébrés 
         * 
         * Precision: tout simplement la précision :P 
            
         FP = Faux positifs 
         * 
         * 
         * La matrice de confusion: 
         * Ligne = classe  réelle, colonne = classe prédite 
         * === Matrice de confusion ===
         * a b c d e f g <-- classified as
         * 5 0 0 0 0 0 0 | a = mammal
         * 0 4 0 0 0 0 0 | b = bird
         * 0 0 2 0 1 0 0 | c = reptile
         * 0 0 0 3 0 0 0 | d = fish
         * 0 0 0 0 2 0 0 | e = amphibian
         * 0 0 0 0 0 2 1 | f = insect
         * 0 0 0 0 0 0 3 | g = invertebrate
         */

    }
}
