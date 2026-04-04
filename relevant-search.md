# Important concepts
  * query DSL
  * recall and precision
  * components of an analyzer
    * important to note that an analyzer defined on a field will be performed on both the doc and the search query
  * tf-idf
  * asymmetric analysis
  * field-centric versus term-centric search 
# Search under the hood
* **documents** as a basic unit in a search engine
  * contain fields
  * ranked according to relevance
  * use *facets* to present summary of results outside the selected criteria (such as filters etc)
* document ingestion
  * original text undergoes a process known as *analysis* to be converted into *tokens*
  * *analysis* is followed by *index*: tokens are ingested into a search engine
* **inverted index**
  * similar to a book's index where terms are linked to the page number
  * consists of a *term dictionary* (numbered list of all terms in the index) and a *posting list* (matching ordinal numbers from the dictionary with a list of documents that contain that term)
  * other data structures in Lucene to facilitate search
    * doc frequency, term frequency, term position, term offsets, payloads, stored fields, doc values
* token matching: both user query and documents are tokenized through analysis, then matched
  * token matches have to be exact
  * tokens are *features* of a document
* **analysis**
  * three steps: character filtering, tokenization and token filtering
  * multiple character and token filters can be used in the analysis process, but only one tokenizer 
  * extra data can be added to tokens:
    * term position: enable phrase matches
    * term offsets: enable result highlighting
    * payload: arbitrary data
  * storing vs indexing:
    * storing enable retrieving, but not searchable
    * only indexed content can be searched, indexed text is saved in the inverted index
* **retrieval**
  * boolean operator like AND and OR combines multiple search results
    * AND operator will find results independently for both queries and gather the docs that are in both results
  * lucene-based boolean queries: MUST, MUST_NOT and SHOULD
    * MUST has to have a match inside a doc
    * SHOULD might or might not have a match; matched documents are ranked higher
    * a BooleanQuery can have any number of SHOULD/MUST/MUST_NOT queries
    * syntax: `black +cat -dog` means a document should have 'black', must have 'cat', and must not have 'dog'
    * if a doc contains a MUST_NOT, it won't be matched even if it has MUST or SHOULD terms
    * lucene's syntax can represent the idea of "should" in a simpler way
    * output of lucene boolean queries are a list of ids
    * scores are often summed up as a list of SHOULD queries
  * term positions with *phrase query*
    * a phrase query is adjacent terms, represented in quotations
    * lucene will fine docs that contain both term, and only retain docs where these terms are adjacent
    * output a list of ids, compatible with boolean queries
    * inverted index by default records term positions, which makes the position check easy
    * *phrase slop* can relax query position restrictions while *span queries* allow more control
  * filtering, facets and aggregations
    * product search based on categories or price are filters on low-cardinality fields
    * search by categories, like the filters on product search pages, are also called *facets*
  * sorting, ranking, relevance
    * sorting can be customized, but by default, it is ranked from most to least relevant
    * a broader definition of relevance include business requirements, on top of accuracy of information retrieved
    * ranking function
      * TF-IDF
        * *term frequency*: how often a term occurs in a doc
        * *doc frequency*: how often a term is in docs within the collection (# of collections containing the term/total # docs); IDF is log of inverse doc frequency
          * IDF is larger for rare words and small fields
        * for a longer query, each term in a field are combined through SHOULD clauses, their scores are combined; not all terms have to present
        * a query is searched across multiple fields, one field can have multiple indices and searched together; can combine them with weights
        * popularity can be used as weights
# Debugging search engine
  * scenario: search "basketball with cartoon alien" and desired result "space jam"
  * term frequency is the # of times a term appears in a index; when an index is divided into shards, term frequency can be affected; the effect is less in larger index compare to smaller ones
  * `multi_match` is a simple way to construct queries across multiple fields
  * to debug unexpected search behavior tackle 2 aspects:
    * query parsing: how your DSL query behave
    * analysis: how your query and docs are broken into tokens
  * `validation` API from elastic search explains query parsing
    * an sample output:
    ```
    “((title:basketball title:with title:cartoon title:aliens)^10.0) |
    (overview:basketball overview:with overview:cartoon overview:aliens)”
    ```
    where `<fieldName>:<query>` represents a term query, a single term lookup
  * **analyzer** specifies how to ingest a document with char fitlers, tokenizer and token filters
    * use `_analyze` endpoint to see how text are parsed
  * use `english` analyzer that include common stop words like "with"
  * document score calculation:
    * think about how search term relate to your features of the doc (eg. can the word "alien" capture cartoon alien feature in the index?)
    * how features related to one another (how important are they across different fields)
  * `explain:true` to output search query
  * vector-space model
    * we view items in an index **vectors**
    * similar items in a vector space have higher dot product; the goal of the present search is to find text that have both "alien-ness" and "basketball-ness"
    * bag of words: every word in English has a dimension of its own, so in a document, most dimensions are empty or *sparse* since it's not possible to have every word represented in a text
    * in semantic search, the comparable items are the user query and a doc in the index
  * **lucene scoring**:
    * not always summing the results of multiplications
    * it can be an OR operator of two groups
    * *coord* punishes compound matches that are incomplete
    * normalization by dividing the magnitude
    * does not make sense to compare scores between queries
    * a term's weight in a field is defined by *similarity*, which is calculated by `tf x idf`
    * lucene dampens the impact of tf with a similarity class
    * *FieldNorm* is a measure of field length: a term match in a shorter text is more important than a match in a longer one; it is calculated as `1 / sqrt(fieldlength)`
    * lucene's classic similarity formula: `tf x idf x fieldNorm`
    * **Okapi bm25 or (bm25)**: more complex tf and fieldNorm calculation
    * search term scores (query): query weight has two components boosting, query normalization
    * query scores don't have term frequency or field norm; 
    * queryNorm makes matches across search results comparable, but it's crude, so don't compare scores across different field (the term 'alien' in overview and in title fields aren't comparable: boost title by 10 doesn't mean title scores are 10 times that of overview)
    * a boost of 10 doesn't mean a field is 10 times more important than another
    * boost can be lost in queryNorm calculations
  * have a boost of 0.1 might still mean this field has a higher value comparing to the other
# Taming tokens
  * relevance is similar to classification, tokens are features of a doc
  * **precision** and **recall**
    * precision: documents that are relevant in the result
      *  # relevant in result/# of total result
      * TP (true positive)/(TP + FP)
    * recall: percentage of relevant documents that are returned
      * # relevant in result/# of total relevant doc
      * TP / (TP + FN)
    * they are often at odds with each other
    * to improve them, add more features
    * the default, standard analyzer is high precision because it has strict requirements for matching (not much varieties in tokens)
  * ideally, analysis don't map words to token; it maps *meaning* and user *intent* to tokens
  * *english_keywords* filter: a list of words that should not be stemmed ("maine" will be stemmed to "main", which is undesirable)
  * **stemming** works by transforming word to root form (not using a dictionary, but thru heuristics)
  * stemming is a technique that sacrifices precision for recall
  * a phonetic analyzer maps words to tokens based on the way it sounds
    * this improves recall but hurt precision since irrelevant words will be mapped to real intent based on phonetics ("from Dalai Lama" and "from tall llama" are mapped to the same tokens)
    * lessons: *To the extent possible, make sure that tokens represent not just words, but the meaning of the words.*
    * token stemming is a way to capture meaning
  * "A search engine is nothing more than a sophisticated token-matching system and document-ranking system."
    * rank top n relevant results in order to balance precision and recall
    * tf-idf controls ranking based on tokens: without stemming, "run" and "running" will be different ideas and scored differently
    * analysis can improve the computation of tf-idf by normalizing the different representations of one idea
  * a strict standard analyzer is already very precise, often too precise in practice
  * The goal of analysis is to loosen the restriction through tokenization strategies like stemming, but this introduces a trade-off between (more) recall and (less) precision
  * To compensate for the trad-off, tweak relevance scores through analysis
  * **coord** will promote docs by reduce scores of docs that only match one word out of several search terms; eg, if the search query is "banana apple", the final score of the doc with only "apple" will be reduced by coord
  * important to note that an analyzer defined on a field will be performed on both the doc and the search query
  * **asymmetric analysis**: analysis at query time is different from analysis at index time
    * for a phrase "dress shoe", produce "dress_shoe" and "shoe" at index time, and only "dress_shoe" at search time, that way, when you search "shoe", you will still get "dress shoe" docs
    * *semantic expansion*: expand terms to equal or greater generality (with synonyms):
    ```
    apple => apple, fruit
    fuji => fuji, apple, fruit
    mcintosh => mcintosh, apple, fruit
    gala => gala, apple, fruit
    banana => banana, fruit
    orange => orange, fruit
    ```
    If this is applied at index time, when you search "apple" or "fruit", you will get "fuji" doc in the results (conversely, if you apply at search time, you get "apple" and "fruit" docs when you look for "fuji")
    * this is particularly useful for specialized field when a taxonomy can be used to map term hierarchy: for example, you can map a med to "anxiety" or "depression" rather than just its name, that way, users will find these meds when they look up more general terms
  * use path to model specificity with *path_hierarchy* analyzer
# multi-field search
  * how to combine scores from individual field into a ranking function
  * *signal*: part of relevant-score calculation that provide insight into user or business information
    * *signal modeling*: control every aspect of relevance scoring so that you approach signals that measure the desired info
    * *manipulate the ranking function*
  * `multi_match` scores
  * example: how to look up "Adam P. Smith" in a index where first name, middle name and last name are separate?
    * `multi_match` for query against 3 fields is inefficient:
    ```
    (first_name:adam first_name:p first_name:smith) |
    (middle_initial:adam middle_initial:p middle_initial:smith) |
    (last_name:adam last_name:p last_name:smith)
    ```
    * the problem here is that the field score provides ambiguous information, the meaning of a `first_name` score is vague since any search term can be first_name
    * the solution is to provide a "full_name" field in addition to the three separate fields so that a full name query can be matched against since user often look up full names, then we use a phrase match on this field
    ```
    max(first_name:Adam first_name:P first_name:Smith),
    (middle_initial:Adam middle_initial:P middle_initial:Smith),
    (last_name:Adam last_name:P last_name:Smith))
    + full_name:"Adam P. Smith
    ```
    the plus is a SHOULD clause, where scores are added together
    here if there is a phrase match score, then the query has an exact match, otherwise, there isn't
    * construct fields to cater to search behavior
* nested fields (managing relations), eg, a list of objects 
  * inner objects: flattens each filed of the object list, but lose relationship between an attribute of an object
* *field-centric* search:
  * two approaches to ranking multiple fields, *field-centric* and *term-centric*
  * Field-centric search runs the search string against each field, combining scores after each field is searched in isolation
  * *Term-centric* search works just the opposite, by searching each field on a term-by-term basis. The result is a per-term score that combines each field’s influence for that term
  * `mutli-match` is an example of field-centric search since query terms go thru the same set of fields in isolation and then combined
  * scoring multiple fields: `best_fields` or `most_fields`
    * `best_fields` by taking the highest score field and add tie breaker: `score = S_title + tie_breaker x (S_overview + S_cast + S_director)`, or simply, max
    * `most_field` like a boolean query, sum matching scores then apply a *coord* factor (# of matching clause/# total clause): `score = (S_overview + S_title + ...) x coord`, or simply, sum
    * use `best` when docs rarely have multiple fields that match search string; use `most` when multiple fields match
    

