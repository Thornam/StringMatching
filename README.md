# StringMatching

The code in this repository is created as a part of the research paper *"Research on the Opportunities for Danish Deep Tech Startups to Get Hard Money in Their Early Development Stages"*, where the opportunity for Danish startup companies to get funding in the venture market was investigated. To do this, we had access to three databases specializing in startup data, where we needed core variables from all three. The challenge in this was to merge information from all three databases without a firm-specific number that could identify the individual firm. 

Therefore, we had to create a system that uses the company names in each database to match company information. This was created by using **N-grams**, **TF-IDF vectorization**, and **Cosine similarity** to get an effective matching of the name strings with a very low percentage of mismatches. 

For an in-depth explanation of the method used in the Python code, one can read the data section from the paper which is added to this repository. 
