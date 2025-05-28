package com.example;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;

import java.util.Scanner;

public class WekaPredictInteractive {
    public static void main(String[] args) throws Exception {
        // Charger la structure du dataset
        DataSource source = new DataSource("src/main/ressources/zoo.arff");
        Instances structure = source.getStructure();

        // Supprimer l'attribut 'animal' s'il est de type STRING
        if (structure.attribute("animal") != null && structure.attribute("animal").isString()) {
            structure.deleteAttributeAt(structure.attribute("animal").index());
        }

        structure.setClassIndex(structure.numAttributes() - 1);

        // Charger le modèle
        Classifier cls = (Classifier) SerializationHelper.read("zoo-model.model");

        // Créer une nouvelle instance vide (après avoir nettoyé la structure)
        Instance instance = new DenseInstance(structure.numAttributes());
        instance.setDataset(structure);

        // GAME! START!, avec un nouveau Scanner on demande...
        // https://www.w3schools.com/java/java_user_input.asp
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("=== Saisie des caractéristiques de l'animal ===");

            // Lire les valeurs des attributs ; on s'arrête avant d'entrer la ligne a
            // predire
            for (int i = 0; i < structure.numAttributes(); i++) {
                Attribute attr = structure.attribute(i);
                if (attr.index() == structure.classIndex()) {
                    continue; // Ne pas saisir la classe
                }

                if (attr.isNominal()) { // Nominal = boolean
                    System.out.print(attr.name() + " (true/false) : ");
                    String val = scanner.nextLine().trim().toLowerCase();
                    while (!val.equals("true") && !val.equals("false")) {
                        System.out.print(" Veuillez entrer 'true' ou 'false' : "); // Validation!!!!
                        val = scanner.nextLine().trim().toLowerCase();
                    }
                    instance.setValue(attr, val); // On set les values
                } else if (attr.isNumeric()) { // Else if pour le nombre de pattes :p
                    System.out.print(attr.name() + " (nombre entier) : ");
                    int val = Integer.parseInt(scanner.nextLine().trim());
                    instance.setValue(attr, val);
                }
            }

            // Prediction ; Cls (notre modele) classify l'instance (qu'on vient de créer!)
            double prediction = cls.classifyInstance(instance);
            // La classe prédite
            String predictedClass = structure.classAttribute().value((int) prediction);

            System.out.println("\n=== Résultat de la prédiction ===");
            System.out.println("Classe prédite : " + predictedClass);

            // Distribution des probabilités
            double[] distribution = cls.distributionForInstance(instance);
            System.out.println("\n=== Probabilités estimées ===");
            // Petite loop qui imprime les probabilité calculées
            for (int i = 0; i < distribution.length; i++) {
                System.out.printf("%-15s : %.2f%%\n", // print format
                        structure.classAttribute().value(i),
                        distribution[i] * 100);
            }

            /*
             * Voici comment lire  ce format "%-15s : %.2f%%\n" :
             * 
             * %-15s
             * 
             * s = tu formates une chaîne (string).
             * 
             * 15 = largeur minimale de 15 caractères.
             * 
             * - = alignement à gauche : si la chaîne fait moins de 15 caractères, elle est
             * complétée par des espaces à droite.
             * 
             * Exemple : "mammal" devient "mammal " (6 lettres + 9 espaces).
             * 
             * :
             * 
             * C’est juste du texte brut : un espace, deux‑points, un espace, pour séparer
             * le nom de la classe de sa probabilité.
             * 
             * %.2f
             * 
             * f = tu formates un nombre à virgule (float/double).
             * 
             * .2 =2 décimales après la virgule.
             * 
             * Exemple : 0.666666 devient 0.67.
             * 
             * %%
             * 
             * Pour afficher un signe % littéral, tu écris %% dans le format.
             * 
             * Résultat : un seul caractère % à l’écran.
             * 
             * \n
             * 
             * Retour à la ligne : après avoir affiché la chaîne + nombre + %, on passe à la
             * ligne suivante.
             */

            // Comparaison facultative avec la vérité :p 
            System.out.print("\nQuelle était la vraie classe ? (optionnel, enter pour ignorer) : ");
            String verite = scanner.nextLine().trim();
            if (!verite.isEmpty()) {
                if (verite.equalsIgnoreCase(predictedClass)) {
                    System.out.println(" Bravo ! Le modèle a deviné correctement !");
                } else {
                    System.out.println(" Oups ! Le modèle s’est trompé.");
                }
            }
        }

        System.out.println("\n=== Fin de l'analyse ===");
    }
}
