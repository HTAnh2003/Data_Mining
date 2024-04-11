package prj_datamining;

import weka.core.Instances;

public class test_ClassificationModel {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		Instances datatrain;
		ClassificationModel clf = new ClassificationModel();
		datatrain = clf.loadData("C:\\Program Files\\Weka-3-8-6\\data\\weather.nominal.arff");
		clf.build_J48Model(datatrain);
		System.out.println(clf.output_J48());
		Instances datatest = datatrain;
		clf.evalJ48Model(datatest,10);
		System.out.println(clf.outputEvalJ48());
		
		clf.build_OneR_Model(datatrain);
		System.out.println(clf.eval_OneR(datatest));
		
		clf.build_NaiveBayesModel(datatrain);
		System.out.println(clf.eval_Naive(datatest));
	}

}
