package prj_datamining;

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class ClusteringModel {
	DataSource source;
	SimpleKMeans kmeans;
	HierarchicalClusterer agnes;
	
	public Instances loadData(String fileName) throws Exception {
		source = new DataSource(fileName);
		return source.getDataSet();
	}
	
	public void build_Cluster_Kmeans(Instances data,int numCluster,boolean isManhattan) throws Exception {
		kmeans = new SimpleKMeans();
		kmeans.setNumClusters(numCluster);
		if(isManhattan)
			kmeans.setDistanceFunction(new ManhattanDistance());
		else
			kmeans.setDistanceFunction(new EuclideanDistance());
		kmeans.buildClusterer(data);
	}
	
	public void build_Cluster_Hierarchical(Instances data,int numCluster,boolean isManhattan,String link) throws Exception {
		agnes = new HierarchicalClusterer();
		agnes.setNumClusters(numCluster);
		if(isManhattan)
			agnes.setDistanceFunction(new ManhattanDistance());
		else
			agnes.setDistanceFunction(new EuclideanDistance());
		
		final int SINGLE = 0;
	    final int COMPLETE = 1;
	    final int AVERAGE = 2;
	    final int MEAN = 3;
	    final int CENTROID = 4;
	    final int WARD = 5;
	    final int ADJCOMPLETE = 6;
	    final int NEIGHBOR_JOINING = 7;
	    
	    SelectedTag typelink;
	    if (link.equalsIgnoreCase("single")) {
	    	typelink = new SelectedTag(SINGLE, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if (link.equalsIgnoreCase("complete")) {
	    	typelink = new SelectedTag(COMPLETE, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if(link.equalsIgnoreCase("average")) {
	    	typelink = new SelectedTag(AVERAGE, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if(link.equalsIgnoreCase("mean")) {
	    	typelink = new SelectedTag(MEAN, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if (link.equalsIgnoreCase("centroid")) {
	        typelink = new SelectedTag(CENTROID, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if (link.equalsIgnoreCase("ward")) {
	        typelink = new SelectedTag(WARD, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if (link.equalsIgnoreCase("adjcomplete")) {
	        typelink = new SelectedTag(ADJCOMPLETE, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else if (link.equalsIgnoreCase("neighbor_joining")) {
	        typelink = new SelectedTag(NEIGHBOR_JOINING, HierarchicalClusterer.TAGS_LINK_TYPE);
	    } else {
	    	// Nếu không phù hợp với bất kỳ giá trị nào, mặc định là sử dụng complete link
	        typelink = new SelectedTag(COMPLETE, HierarchicalClusterer.TAGS_LINK_TYPE);
	        System.out.println("Giá trị của link không hợp lệ. Đã thiết lập mặc định là complete link.");
	    }
	    agnes.setLinkType(typelink);
		agnes.buildClusterer(data);
	}
	
	public void predict_Cluster_Label(Instances data) throws Exception {
		System.out.println("-----------PREDICT CLUSTER LABEL-------------------");
		for(Instance inst: data) {
			int num = kmeans.clusterInstance(inst);
			System.out.println(inst.toString() + " thuoc cum : " + num);
		}
	}
	
	public String output_Kmeans() {
		return kmeans.toString();
	}
	
	public void predict_Hieracchical(Instances data) throws Exception {
		System.out.println("-----------PREDICT Hieracchical-------------------");
		for(Instance inst:data) {
			int num = agnes.clusterInstance(inst);
			System.out.println(inst.toString() + " thuoc cum : " + num);
		}
	}
	public String output_Hieracchical() {
		System.out.println("-----Output Hieracchical--------");
		return agnes.toString();
	}
}
