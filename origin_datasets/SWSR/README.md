# README

## Overview

Our repository presents the Sina Weibo Sexism Review (SWSR) dataset containing sexism-related posts in Chinese collected from Sina Weibo, as well as the Chinese lexicon SexHateLex. The SWSR dataset can be exploited for building computational methods to identify and investigate online, gender-related abusive language. The SexHateLex lexicon can also support detection and analysis of sexist contents.


## Data Format

SWSR dataset consists of two files:  `SexWeibo.csv` and `SexComment.csv`, corresponding to 1527 weibos and 8969 comments. The *SexHateLex* lexicon contains a list of 3016 abusive terms in `SexHateLex.txt`. We only show the format of SWSR dataset here.

#### SWSR Dataset

* **SexWeibo.csv**

    * weibo_id: a string of weibo ID
    * weibo_text: a string of weibo text
    * keyword: contains  the  sexist  keyword(s)  extracted  from  the weibo. Not every weibo has corresponding keyword(s) 
    * user_gender: the gender of user
    * user_location: the location of user
    * user_follower: number of users who follow this user's account
    * user_following: number of users whom this user follows
    * weibo_like: number of like for the weibo
    * weibo_comment: number of comment for the weibo
    * weibo_repost: number of repost for the weibo
    * weibo_date: the date and time when the weibo is posted
                

* **SexComment.csv**

    * weibo_id: the weibo id where the comment is collected
    * comment_text: a string of comment text 
    * gender: the gender of commenter
    * location: the location of commenter
    * like: number of like for this comment
    * date: the date and time when the comment is posted
    * label: the comment is sexist(1) or non-sexist(0)
    * category: categorise sexism into four classes -- Stereotype based on Appearance(SA), Stereotype based on Cultural Background (SCB), MicroAggression (MA) and Sexual Offense (SO)
    * target:  the type of target who are attacked -- Individual (I) or Generic (G)


## Reference

A. Jiang, X. Yang, Y. Liu and A. Zubiaga (2021). SWSR: A Chinese Dataset and Lexicon for Online Sexism Detection. *Under review*

   
   



