## Scale
### horizontal
* load balancer: distribute incoming requests across servers
    * tool: nginx
* assumptions: 
    * servers are stateless, request from the same user can be handled by different servers
    * database is shared among the servers
* database strategies:
    * sharding: spread database over db instances
* resources: http://web.stanford.edu/class/archive/cs/cs142/cs142.1196/lectures/LargeScaleWebApps.pdf
### vertical
* https://github.com/donnemartin/system-design-primer from (https://twitter.com/ryanlpeterman/status/1779537662271246500)
## Content Distribution Network (CDN)
* store content to servers positioned all over the world for faster content loading
* mechanism:
    * you give some content and they give you a URL
    * put that URL in your app
    * when user accesss that URL, they are taken to the closest server (thru DNS tricks)
* a read-only part of the web app
* only works on content that doesn't change often
## Network
* build server
* implement protocols
* browser related
* secure shell