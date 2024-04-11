package prj_datamining;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificationModel {
	DataSource source;
	J48 j48;
	OneR rules;
	NaiveBayes nb;
	Evaluation eval;
	
	public Instances loadData(String fileName) throws Exception {
		source = new DataSource(fileName);
		return source.getDataSet();
	}
	
	public void build_J48Model(Instances datatrain) throws Exception {
		j48 = new J48();
		datatrain.setClassIndex(datatrain.numAttributes()-1);
		j48.buildClassifier(datatrain);
	}

	public void build_OneR_Model(Instances datatrain) throws Exception {
		rules = new OneR();
		datatrain.setClassIndex(datatrain.numAttributes()-1);
		rules.buildClassifier(datatrain);
	}
	
	public void build_NaiveBayesModel(Instances datatrain) throws Exception {
		nb = new NaiveBayes();
		datatrain.setClassIndex(datatrain.numAttributes()-1);
		nb.buildClassifier(datatrain);
	}
	
	public String outputEvalJ48() throws Exception {
		return eval.toSummaryString()+"\n"+eval.toMatrixString()+"\n"+eval.toClassDetailsString();
	}
	
	public void evalJ48Model(Instances datatest,int k) throws Exception {
		datatest.setClassIndex(datatest.numAttributes()-1);
		eval = new Evaluation(datatest);
//		eval.evaluateModel(j48, datatest);
		Random rd = new Random();
		rd.setSeed(1);
		eval.crossValidateModel(j48, datatest, k, rd);
	}
	public String eval_OneR(Instances datatest) throws Exception {
		System.out.println("------Evaluation OneR--------");
		datatest.setClassIndex(datatest.numAttributes()-1);
		Evaluation evalOneR = new Evaluation(datatest);
		evalOneR.evaluateModel(rules, datatest);
		return evalOneR.toSummaryString();
	}
	
	public String eval_Naive(Instances datatest) throws Exception {
		System.out.println("------Evaluation Naive Bayes--------");
		datatest.setClassIndex(datatest.numAttributes()-1);
		Evaluation evalNaive = new Evaluation(datatest);
		evalNaive.evaluateModel(nb, datatest);
		return evalNaive.toSummaryString();
	}
	public String output_J48() {
		return j48.toString();
	}
}
