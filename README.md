# reviewclassification

Review classification (positive/negative)

JAVA Dependencies:

1. Lucene: http://www.apache.org/dyn/closer.lua/lucene/java/6.4.1
2. JSON: https://search.maven.org/remotecontent?filepath=org/json/json/20160810/json-20160810.jar

For Lucene, you need to add 2 JAR's:

1. lucene-6.2.1/core/lucene-core-6.4.1.jar
2. lucene-6.2.1/queryparser/lucene-queryparser-6.4.1.jar

Alternatively, you can use the pom.xml file with Maven to take care of JAVA dependencies.

Python Dependencies:

1. BS4 - pip install bs4
2. NTLK - pip install nltk
3. Python interpreter - nltk.download() -> Models -> punkt -> Download
4. Python interpreter - nltk.download() -> Models -> averaged_perceptron_tagger -> Download
