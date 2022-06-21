import requests
from lxml import etree
import json

headers = {
    'Accept': '*/*',
    'Connection': 'keep-alive',
    'Cookie': 'SINAGLOBAL=7547742164796.669.1557814374521; wvr=6; UOR=vjudge.net,widget.weibo.com,www.baidu.com; Ugrow-G0=d52660735d1ea4ed313e0beb68c05fc5; login_sid_t=5355b1f41984fbcab1e7f06ce5e56348; cross_origin_proto=SSL; TC-V5-G0=28bf4f11899208be3dc10225cf7ad3c6; WBStorage=f54cf4e4362237da|undefined; _s_tentry=passport.weibo.com; Apache=137824208234.45386.1567069036053; ULV=1567069036066:8:4:2:137824208234.45386.1567069036053:1566997217080; wb_view_log=1366*7681; SCF=Ah0xAfGqEnBKvHeo0pUZh35Mu9kMRIv3xACyV4NhxHz6vEmrCbVYAZNMkq28jwwjv3izlLBbVs0-1wHJgB802KI.; SUB=_2A25wY-PeDeRhGeBH6lcQ9SrMyjmIHXVTGVIWrDV8PUNbmtAKLUfgkW9NQcb_zmmXRFBtE4vRVttRPlpfY4_u__na; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5PjVhlq3PienOo1MfYibdR5JpX5KzhUgL.Foq4eK-pSKB7eK-2dJLoIpzLxKMLBK-LBKBLxKqLBo.LB-zt; SUHB=0KAhxcJ_lXGMkr; ALF=1598605070; SSOLoginState=1567069071; wb_view_log_6915154015=1366*7681; webim_unReadCount=%7B%22time%22%3A1567069360213%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22allcountNum%22%3A0%2C%22msgbox%22%3A0%7D; TC-Page-G0=ac3bb62966dad84dafa780689a4f7fc3|1567069357|1567069073',    'Host': 'weibo.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
