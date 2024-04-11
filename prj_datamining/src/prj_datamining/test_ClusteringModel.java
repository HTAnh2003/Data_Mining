package prj_datamining;

import weka.core.Instances;

public class test_ClusteringModel {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ClusteringModel cluster = new ClusteringModel();
		Instances data = cluster.loadData("C:\\Program Files\\Weka-3-8-6\\data\\weather.nominal.arff");
		cluster.build_Cluster_Kmeans(data,7,false);
		System.out.println(cluster.output_Kmeans());
		cluster.predict_Cluster_Label(data);
		
		cluster.build_Cluster_Hierarchical(data, 7, false, "single"); 
		System.out.println(cluster.output_Hieracchical());
		
		cluster.build_Cluster_Hierarchical(data, 7, false, "complete"); 
		System.out.println(cluster.output_Hieracchical());
		
		cluster.build_Cluster_Hierarchical(data, 7, false, "average"); 
		System.out.println(cluster.output_Hieracchical());
		
		cluster.build_Cluster_Hierarchical(data, 7, false, "mean"); 
		System.out.println(cluster.output_Hieracchical());
		
		cluster.build_Cluster_Hierarchical(data, 7, false, "centroid"); 
		System.out.println(cluster.output_Hieracchical());
		
		cluster.predict_Hieracchical(data);
	}

}
