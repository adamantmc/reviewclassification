/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reviewclassification;

import java.io.File;
import java.io.IOException;
import static java.lang.Math.log;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.json.JSONObject;

/**
 *
 * @author swagdam
 */
public class ReviewClassification {

    public static String replaceRegex = "[^a-zA-Z ]+";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, ParseException, org.apache.lucene.queryparser.classic.ParseException {
        SimpleDateFormat ft = new SimpleDateFormat("hh:mm:ss");
        String base_dir = "processed_data_all_scored";
        ArrayList<Document> documents = readReviews(base_dir+"/train/neg/", -1);
        documents.addAll(readReviews(base_dir+"/train/pos/", -1));

        long seed = System.nanoTime();  
        //Shuffle the documents, so that each time the training and test sets are different
        Collections.shuffle(documents, new Random(seed));

        Metrics metrics = new Metrics();

        //9/10 of the training set is used for training
        //The remaining 1/10 is used for testing
        ArrayList<Document> training_set = new ArrayList<>(documents.subList(0, documents.size() * 9 / 10));
        ArrayList<Document> test_set = new ArrayList<>(documents.subList(documents.size() * 9 / 10, documents.size()));

        System.out.println("Total size: " + documents.size());
        System.out.println("Training set size: " + training_set.size());
        System.out.println("Test set size: " + test_set.size());

        ArrayList<Integer> threshold_list = new ArrayList<>();
        ArrayList<Metrics> metrics_list = new ArrayList<>();

        int threshold_start = 3;
        int threshold_end = 100;
 
        for(int i = threshold_start; i <= threshold_end; i++) {
            threshold_list.add(i);
            metrics_list.add(new Metrics());
        }

        BooleanQuery.setMaxClauseCount(100000);

        //Set tf-idf formulas
        Similarity cos_sim = new ClassicSimilarity() {};
        
        
        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = new RAMDirectory();

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setSimilarity(cos_sim);
        IndexWriter w = new IndexWriter(index, config);

        addDocuments(w, training_set);

        w.close();

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(cos_sim);
        
        tlog(ft, "Starting.");
        //For each review in the test set, query the index, get the results, then predict with a given threshold.
        //Testing for multiple thresholds to find which one to use.
        for (Document doc : test_set) {
            ArrayList<Result> results = query(doc, analyzer, searcher, threshold_list.get(threshold_list.size() - 1));
            
            for(int i = 0; i < threshold_list.size(); i++) {
                predict(results, doc, threshold_list.get(i), metrics_list.get(i));
            }
        }

        tlog(ft, "Done.");

        for(int i = 0; i < threshold_list.size(); i++) {
            metrics_list.get(i).calculate();
            System.out.println(threshold_list.get(i) + " " + metrics_list.get(i).getAccuracy());
        }
    }
    
    /**
     * 
     * @param results The results of the query.
     * @param doc The review we want to classify (the query doc).
     * @param threshold The threshold we will use.
     * @param metrics Holds the results of every prediction (tp, tn, fp, fn), we will use it later to calculate the accuracy.
     */
    public static void predict(ArrayList<Result> results, Document doc, int threshold, Metrics metrics) {
        int pos = 0;
        double knn_score = 0.0;
        //double query_score = 0.0, pos_score = 0.0, neg_score = 0.0;
        boolean predicted_positive = false;
        
        for(int j = 0; j < threshold; j++) {
            //If the result is a positive review, we increment the pos variable.
            if(results.get(j).getDoc().get("path").contains("pos")) {
                pos++;
            }
        }
        
        //Calculating the score of the query based on the SentiWordNet dictionary (query_score)
        //As well as the ratio of positive reviews in the results (knn_score)
        //pos_score = doc.getField("pos_score").numericValue().doubleValue();
        //neg_score = doc.getField("neg_score").numericValue().doubleValue();
        //query_score = (double) (pos_score - neg_score) / (pos_score + neg_score);
        
        //Converting knn_score from [0,1] to [-1,1]
        knn_score = 2*((double) pos/threshold) - 1;
        
        //The path of the review we want to classify. This path contains the word "pos" or "neg", which is the class of the review.
        String doc_path = doc.get("path");
        
        //Used to calculate a score based on the knn_score as well as a number indicating the "positiveness" or "negativeness" of the document.
        //Not used, because the results are pretty much the same with knn_score.
        //if(query_score*0.2 + knn_score * 0.8 > 0) predicted_positive = true;
        if(knn_score > 0) predicted_positive = true;
        
        if (predicted_positive) {
            //Predicted positive
            if (doc_path.contains("neg")) {
                //False positive
                metrics.incFp();
            } 
            else {
                //True positive
                metrics.incTp();
            }
        } 
        else {
            //Predicted negative
            if (doc_path.contains("neg")) {
                //True negative
                metrics.incTn();
            } else {
                //False negative
                metrics.incFn();
            }
        }
    }
    
    /**
     * 
     * @param doc The review we want to classify
     * @param analyzer Lucene's analyzer, needed to construct a query
     * @param searcher Lucene's searcher, searches in the index
     * @param threshold K, for top-k results.
     * @return A list with the results of the query.
     * @throws org.apache.lucene.queryparser.classic.ParseException
     * @throws IOException 
     */
    public static ArrayList<Result> query(Document doc, StandardAnalyzer analyzer, IndexSearcher searcher, int threshold) throws org.apache.lucene.queryparser.classic.ParseException, IOException {
        //Create a query after escaping some special characters
        String querystr = doc.get("text").toLowerCase().replaceAll(replaceRegex, "");
        Query q = new QueryParser("text", analyzer).parse(QueryParser.escape(querystr));
        
        //The returned list
        ArrayList<Result> results = new ArrayList<>();
        
        //Search for the query with a given threshold, and get the results in a ScoreDoc array
        TopDocs docs = searcher.search(q, threshold);
        ScoreDoc[] hits = docs.scoreDocs;
             
        //Add the results to the list and return
        for (int i = 0; i < hits.length; i++) {
            results.add(new Result(searcher.doc(hits[i].doc), hits[i].score));
        }

        return results;
    }

    public static String readFile(String path, Charset encoding) throws java.io.IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }

    public static void addDocuments(IndexWriter w, ArrayList<Document> documents) throws IOException {
        for (Document doc : documents) {
            w.addDocument(doc);
        }
    }

    public static ArrayList<Document> readReviews(String directory, int limit) throws IOException {
        ArrayList<Document> documents = new ArrayList<>();

        File dir = new File(directory);
        File[] files = dir.listFiles();

        for (File f : files) {
            String text = readFile(f.getAbsolutePath(), Charset.forName("UTF-8"));
            
            JSONObject jobj = new JSONObject(text);
            Document doc = new Document();
            doc.add(new StringField("path", f.getAbsolutePath(), Field.Store.YES));
            doc.add(new TextField("text", jobj.getString("text"), Field.Store.YES));
            doc.add(new DoublePoint("pos_score", jobj.getDouble("pos_score")));
            doc.add(new DoublePoint("neg_score", jobj.getDouble("neg_score")));

            documents.add(doc);
        }

        return documents;
    }

    public static void tlog(SimpleDateFormat ft, String str) {
        System.out.println("[" + ft.format(new Date()) + "] " + str);
    }

}
