****# Search under the hood
* **documents** as a basic unit in a search engine
  * contain fields
  * ranked according to relevance
  * use *facets* to present summary of results outside the selected criteria (such as filters etc)
* document ingestion
  * original text undergoes a process known as *analysis* to be converted into *tokens*
  * *analysis* is followed by *index*: tokens are ingested into a search engine
* **inverted index**
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