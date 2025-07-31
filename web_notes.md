# full-stack
## cookie and sessions
* User authentication system: https://gist.github.com/siscia/5ed3277551370df3eb8b1063923621d4
* where to store tokens? https://web.archive.org/web/20200710185631/https://stormpath.com/blog/where-to-store-your-jwts-cookies-vs-html5-web-storage
* Why are we using token-based authentication over cookies? https://www.reddit.com/r/webdev/comments/15fgwq5/why_are_we_using_tokenbased_authentication_over/
```
Cookies are 100% the way to go if you have a simple 1 to 1 client server relationship. It's straight forward and easier to make secure but... This server client direct relationship isn't always the case In bigger more complex enterprise architecture.

JWT authentication can be more flexible and is good in situations where you have multiple APIs, multiple different pages or mobile apps using the same APIs.

For example. A large company might have 8 different APIs maintained by different teams written in different languages. A company like this will roll one authentication server that deals with all the Auth and dishes out all the tokens. That means once you have a token you have a common interface to communicate with all the APIs securely.

There are patterns that involve an API gateway that acts as the front door to all of your APIs. Cookie based is very limiting in what you can achieve. JWT means that pretty much anything can call your APIs securely not just browsers.

Cookies don't work in Native apps too. The process you are used to with the apps on your phone keeping you logged in is a JWT refresh Oauth flow. It would mean maintaining an Auth server on top of a cookie based Auth flow for your webpage. Which doesn't make sense so you just use JWT and either Oauth/OpenID etc for everything. 
```
### cookies for data storage
* From [MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies):
> In the early days of the web when there was no other option, cookies were used for general client-side data storage purposes. Modern storage APIs are now recommended, for example the Web Storage API (localStorage and sessionStorage) and IndexedDB. They are designed with storage in mind, never send data to the server
that is because: 1, cookie # and sizes are limited, while Storage API has bigger capacity; 2, cookies are sent with every request, so they slow down performance
* read this [article](https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Client-side_APIs/Client-side_storage) for more on client side sotrage
## CORS
* mdn definition: Cross-Origin Resource Sharing (CORS) is an HTTP-header based mechanism that allows a server to indicate any origins (domain, scheme, or port) other than its own from which a browser should permit loading resources. 
* in short, CORS is a mechanism that prohibit browser to load content if server prohibits it
* CORS does not prevent server from receiving request from urls
* CORS is not a solution against all types of cross-site scripting (XSS): 
> TL;DR: How does CORS prevent XSS? It does not. It is not meant to do so ... ORS is intended to allow resource hosts (any service that makes its data available via HTTP) to restrict which websites may access that data ... CORS is unrelated to XSS because any attacker who can place an evil piece of JavaScript into a website can also set up a server that sends correct CORS headers. CORS cannot prevent malicious JavaScript from sending session ids and permlogin cookies back to the attacker. 
from https://security.stackexchange.com/questions/108835/how-does-cors-prevent-xss
## Image
### webp image
* https://www.reddit.com/r/webdev/comments/1dz55ww/anyone_switched_over_to_primarily_using_webp/
### image loading
* using small image size
* use `loading="lazy"` html tag to enable lazy loading
* use a placeholder image (blurhash, ffmpeg)
	* there are many ways to implement a placeholder image, nextjs uses a css background image: https://github.com/vercel/next.js/blob/canary/packages/next/src/shared/lib/get-img-props.ts#L723C30-L723C47 
* show image when fully loaded through (detect `complete` property of `img`)
* use src and sizes to load smaller images for smaller devices
* source: https://blog.webdevsimplified.com/2023-05/lazy-load-images/ and https://archive.is/Yae7m (https://medium.com/programming-essentials/how-to-detect-when-all-images-are-loaded-in-a-react-component-d831d0c675b2) and https://ericportis.com/posts/2014/srcset-sizes/
# API Design
* API Design best practices from Microsoft https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design
* Learn rest api design: https://www.restapitutorial.com/
## Webhook
* Webhook is a request (usually POST and GET) from a service provider (eg. Stripe) to your application. It is a way to feed information back to application from the provider (eg. transaction complete).
* Alternative solutions are short and long pooling. Long pooling is when the provider opens a port and wait for the process to finish before sending the request to the application, but this method consumes more server resources of the provider than webhooks since the it needs to hold the port open.
* cors
	* CORS only affects browser requests, not server-to-server webhook calls
	* Webhook providers (Clerk, Stripe) bypass CORS entirely since they're server-side calls
	* webhook security depends on signature verification
# Web Crawling
* websites have many means of detecting bots, bypassing bots checks require
	* a headless browser like `playwright` or `puppeteer`
	* setting appropriate "User Agent"
* PWA like RECAPTCHA is a thorny issue to overcome
* respect rate limit! otherwise your IP is going to be banned from accessing the site. Check if you scraper has back off/delay mechanism when it hits the rate limit warning
* use site map to get a full list of urls in the website and work from there
* check `robots.txt` to see if the domain can be crawled
### Scrapy
* I used Scrapy to build a crawler to extract some ecommerce data, but my crawler was extremely slow. I ended up switching to `request` to extract data from their backend apis. After adding a realistic header, I was able to finally extract the desired information
* scrapy is more suited for scraping specific sites, another type of crawler is "broad crawl" that targets a large number of domains
* broad crawler
	* does not stop when finished with a domain
	* concurrency
	* don't have much data processing rules (data is post-processed)
	* sample project: https://www.reddit.com/r/rust/comments/ns8vi1/meet_crusty_fast_scalable_polite_broad_web/
### AI applications
* despite its name, `crawl4ai` works very well for ai projects where you need clean markdown files.
* it is really good for single page or recursive crawling for a whole domain, you can restrict the crawler to a single domain
* use the `text_only` option to only extract text
* include or exclude by html tags and css selectors
* I also tried several other options: `jina`, `firecrawl`, and `tavily`. Among them, `tavily` has the worst scraping quality. `firecrawl` is only but the docs are lacking. `jina` is good, but there are less knobs to work with.
* use streaming option to output the result when crawling a large site to avoid memory overflow
## scrapy
* uses spider to collect data from target site
* changing User Agent is helpful, but `scrapy-fake-useragent` package is probably broken
* scrapy doesn't load javascript, would need selenium for that
* use the scrapy shell for interactive sessions
* leverage browser dev tool to verify css selector 
# Programming language
## Go
* Simplicity is Complicated
# Tools
## GraphQL
* GraphQL is a query language for your API, and a server-side runtime for executing queries using a type system you define for your data (https://graphql.org/learn/)
* Think of it as a structured way of making REST calls: these queries are specific to your application, not to the database.
* GraphQL server defines db connections, schema, resolving operation with resolvers (apollo example): https://www.apollographql.com/docs/apollo-server/getting-started
* In an application query language, you ask for things like "Person in age group teens" and not "SELECT * FROM people WHERE age > 12 AND age < 20". The former is how clients ask for data from the server, and the latter is how the server might fetch that data - and clients don't need to know about that. (https://medium.com/apollo-stack/how-do-i-graphql-2fcabfc94a01)
* mutation and query: https://graphql.org/learn/queries/#operation-name https://www.apollographql.com/docs/apollo-server/schema/schema#the-query-type

## Postgres
* postgres consists of a server which runs the database and a client which is invoked with `psql`: see https://www.postgresql.org/docs/current/tutorial-arch.html
* commands on postgresql is not necessarily the same as one installed on prebuilt package
* To start `psql`, start the psql server and use the proper role (usually with username `psql`)
    * these two corresponds to two common error msgs: https://www.postgresql.org/docs/15/tutorial-createdb.html
    * https://www.digitalocean.com/community/tutorials/how-to-install-postgresql-on-ubuntu-20-04-quickstart
    * https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-database
* To create a db with postgres, the db needs to sit in the psql server under the right user with the appropriate role, to do that:
    * the default user for postgres is `postgres`, to create another user switch to `postgres` then create a user with `su postgres`, then `creteuser <username>`
    * might need to alter the new user's priviledge in psql: open `psql` then `alter user <name> createdb`
    * swich to the new user to create new db: `su jli`, then `createdb <dbname>`
    * populate db `psql --dbname <dbname> -f <db.sql>`
    * open server to verify and use `dt` to list all tables in db: `psql` -> `\c <dbname>`  -> `\dt` q
    * Reference: https://stackoverflow.com/questions/43734650/createdb-database-creation-failed-error-permission-denied-to-create-database;
    * https://www.postgresqltutorial.com/postgresql-administration/postgresql-show-tables/
* To start psql
	* check if the server is running: `sudo service postgresql status`
	* start postgresql: `sudo service postgresql start`
	* Connect server and open psql shell with user name `postgres`: `sudo -u postgres psql`
	* list all db: `\l`
	* connect to db: `\c <dbname>`
	* list tables: `\dt`
	* list tables under schema: `\dt schema.*`
* read a file in db: `psql -U jli -d signals -a -f signals_v2.sql`
* show conf within psql: `show config_file` OR `SHOW hba_file`
## Docker
* Don't run docker as root, this will create problem for WSL
    * WSL does not run as root
    * docker defaults to run as root
    * therefore, if docker run as root, you won't be able to edit files docker created
    * https://forums.docker.com/t/editing-files-created-by-a-container/125028/1
    * https://pythonspeed.com/articles/root-capabilities-docker-security/
	* docker logs --follow <container ID>
* docker debugging
	* container may exit if it is not setup correctly, restart the container and debug inside
* port forwarding
	* server inside the container listens on port A, the docker exposes port B, then there is a mismatch
	* docker needs to expose the same port the server is listening, A, in this case
* docker prune -a
## SSL
* certifications need network access
* TLS encrypts DNS queries to keep them secure and private
* It's worth noting that DNS lookup can be sent over TLS/SSL or HTTPS, the former uses port 853, and the latter 443 (https://www.cloudflare.com/en-ca/learning/dns/dns-over-tls/)

## DNS
* domain name can have unlimited number of subdomains subdomains.domain.xxx
* record types:
	* A record: holds IP address (eg. associate a server's IP address with subdomain of your website)
	* AAAA record: IPv6 address
	* CNAM: forwards one domain or subdomain to another, no IP involved
## Spring Boot
### REST API with jpa and hibernate
* Code structure, from https://www.twilio.com/blog/create-rest-apis-java-spring-boot
* testing and mocking in Spring boot https://reflectoring.io/unit-testing-spring-boot
* testing webclient https://www.dontpanicblog.co.uk/2022/01/15/testing-spring-reactive-webclient/
```
DAO - The DAO (data access layer) provides an interface to connect with the database and access the data stored in the database. A single DAO class can deal with queries retrieving different types of entities.
Repository - This layer is similar to the DAO layer which connects to the database and accesses the data. However the repository layer provides a greater abstraction compared to the DAO layer. Every class is responsible for accessing and manipulating one entity. This tutorial will use the repository layer.
Service - This layer calls the DAO layer to get the data and perform business logic on it. The business logic in the service layer could be - performing calculations on the data received, filtering data based on some logic, etc.
Model - The model contains all the Java objects that will be mapped to the database table using. The DAO will fetch the data from the database and populate the respective model with that data and return it to the service layer and vice versa.
Controller - This is the topmost layer, called when a request comes for a particular REST API. The controller will process the REST API request, calls one or more services and returns an HTTP response to the client.
```
* JPA smart implementations based on method names https://docs.spring.io/spring-data/jpa/docs/current/reference/html/#jpa.sample-app.finders.strategies
* Spring @Component annotation
https://www.baeldung.com/spring-component-annotation
* Spring beans
https://stackoverflow.com/a/49443630
* Spring @Autowired
Dependency Injection
https://stackoverflow.com/a/1638976
@Autowired
https://stackoverflow.com/a/19419296
* Enum: treat enum as string: https://stackoverflow.com/questions/6667243/using-enum-values-as-string-literals

## K8S
* clusters
* control plan nodes (must have multiple copies) & worker nodes & pod
* kubelet & containers & worker node
### Concepts
* Minikube is a lightweight Kubernetes implementation that creates a VM on your local machine and deploys a simple cluster containing only one node
	* minikube runs the master processes and worker processes on the same node
	* minikube runs as a docker or a VM
* Kubectl uses the Kubernetes API to interact with the cluster
* Once you have a running Kubernetes cluster, you can deploy your containerized applications on top of it. To do so, you create a Kubernetes Deployment. The Deployment instructs Kubernetes how to create and update instances of your application. Once you've created a Deployment, the Kubernetes control plane schedules the application instances included in that Deployment to run on individual Nodes in the cluster.
	* deployments manages stateless pods
	* to manage stateful pod like DB, use statefulSet instead of deployment (DB are often managed outside of K8S)
*  A Pod is a Kubernetes abstraction that represents a group of one or more application containers (such as Docker), and some shared resources for those containers.
	* more about pod: https://kubernetes.io/docs/concepts/workloads/pods/
*  A Node can have multiple pods, and the Kubernetes control plane automatically handles scheduling the pods across the Nodes in the cluster. A Node is a worker machine in Kubernetes and may be either a virtual or a physical machine, depending on the cluster.
* A Service in Kubernetes is an abstraction which defines a logical set of Pods and a policy by which to access them.
	* each Pod on the same Node has unique and permanent IP address, service expose applications in the pod to the outside and ensure different replica functions the same way (service is also a load balancer)
	* A Service routes traffic across a set of Pods. Services are the abstraction that allows pods to die and replicate in Kubernetes without impacting your application.
	* Even if a pod dies, the new pod get assigned to the same IP address
	* external or internal services
	* Ingress: enable security protocol and domain name for external services
* Terms: https://kubernetes.io/docs/tutorials/kubernetes-basics/explore/explore-intro/
* configMap: externalize config (non-secret) in key value pair files (for credentials, use secrets)
* Volume (data persistence): remote storage or local machine; Kubernetes doesn't manage data persistence
* k8s Configuration File: yaml or json file to configure resource (deployment, service, etc)
* namespaces: they provide a scope for names. Names of resources need to be unique within a namespace, but not across namespaces. 
	* Namespaces are intended for use in environments with many users spread across multiple teams, or projects
	* one cluster can have many deployments, namespaces can be used to distinguish them
	* Can also set limit on resources per NS
	* Namespaces are a way to divide cluster resources between multiple users
	* default namespaces: kube-public (configMaps)
	* examples: monitoring, elastic stack, nginx-ingress OR by teams
	* create namespaces with ConfigMap is a best practice
* ingress:
* helm:
	* template file to set dynamic values
	* Use case: deploy commonly used services, such as elastic stack, involve creating statefulSet, configMap, services, etc. These can be simpliied with helm chart.
* volume: 
	* K8S doesn't manage storage (think of it as external plugin to the cluster)
	* requirements (local storage in the cluster doesn't satisfy requirement 2 and 3)
		* storage can't dependent on pod's lifecycle
		* storage is available for all pods
		* storage should persists even if the cluster crushes
	* persistent volume, persistent volume claim, storage class
		* PersistentVolume: needs actual physical storage (local disk in the cluster, cloud or remote storage) (not namespaced)
		* PersistentVolumeClaim (pvc): applications need to claim storage resourcein PersistentVolume
		* storage class provisions PersistentVolume dynamically whenever PersistentVolumeClaim claims it
			* provisioner manages the allocation
	* remote storage is always preferred over loccal storage
	* volume and volumeMounts: https://youtu.be/X48VuDVv0do?si=jVdTcMnsw3i4BlxQ&t=10482
		* pod specificies what volumes to provide with `spec.volume` and where to mount those in the containers in `spec.containers.volumeMounts`
* statefulSet
	* example: storage/application with storage
	* stateful applications are deployed with StatefulSet; stateless applications are deployed with Deployment
		* both manage Pods based on container specification, configure storage the same way
	* replications of stateful application pods are more difficult
		* each pod are not identical, needs to manage id for each additional pod
	* provision dedicated PersistentVolume for each statefulSet
		* PersistentVolume also has the state of its pod
	* statefulSet's sticky identiy is maintained through DNS names of service and pods
	* in general stateful applications are not suitable for containerized environments, but stateless applications are
	
## Auth
### OAuth
* definition: OAuth 2.0, which stands for “Open Authorization”, is a standard designed to allow a website or application to access resources hosted by other web apps on behalf of a user.
* components: client, API (Resource Server), Authorization Server, User (Resource Owner)
	* the client is the application requesting user's account (this is the website you are trying to register with your google account)
	* API server that has access to user info (think about Google server that has your accounts)
	* Authorization server: it presents the interface that asks user to approve or deny access (the popup window); in smaller implementations, it is the same server as the API
* Steps:
	1. Authrization server asks for approval from the user
	2. user approves, gets redicted to the application interface with a Authorization Code
	3. user exchanges the Authorization Code with a access token from the Authorization Server
	4. user can use this access token to access their information on the Resource Server
* https://developers.google.com/oauthplayground/
* Protocol flow: https://datatracker.ietf.org/doc/html/rfc6749#section-1.2

     +--------+                               +---------------+
     |        |--(A)- Authorization Request ->|   Resource    |
     |        |                               |     Owner     |
     |        |<-(B)-- Authorization Grant ---|               |
     |        |                               +---------------+
     |        |
     |        |                               +---------------+
     |        |--(C)-- Authorization Grant -->| Authorization |
     | Client |                               |     Server    |
     |        |<-(D)----- Access Token -------|               |
     |        |                               +---------------+
     |        |
     |        |                               +---------------+
     |        |--(E)----- Access Token ------>|    Resource   |
     |        |                               |     Server    |
     |        |<-(F)--- Protected Resource ---|               |
     +--------+                               +---------------+
* validate jwt token example from Cube https://github.com/cube-js/cube/blob/401e9e1b9c07e115804a1f84fade2bb82b55ca29/packages/cubejs-api-gateway/src/gateway.ts#L2047C43-L2047C46
* Client Credentials Flow，this is the workflow where authorization server issues access token based on client identity (ID, secret). The client is registered in authorization server in this case. https://auth0.com/docs/get-started/authentication-and-authorization-flow/client-credentials-flow
### sessions
* A session represents an instance in which a user has signed in and is authenticated. Normally, when a user signs in successfully, a session is created and typically stored in a database, along with the user's ID, a status (active, revoked, expired, etc) and an expiration time, at miniumum. The session must be active and not expired in order for the user to be seen as authenticated. One user can, and typically does, have multiple sessions. Each time the user signs in, whether on the same or a different device, creates a new session (from Clerk docs: https://clerk.com/docs/how-clerk-works/overview)
* one misconception about session is token-based auth does not deal with sessions. This is wrong. Token-based auth just not saving the session information in the db anymore. Sessions still need to be managed. Sessions is a more generic concept than any implementation models. It refers to episodes of interaction where the user is validated. For instance, managing sessions in token-based auth refers to keep track of when user signs out and when the token expires to prevent active session even after the user signs out
#### session-based (stateful)
* The session is stored in a cookie and is sent to the server on every request. The server then verifies the session and returns the user data if the session is valid
* once validation succeeds, server sends back a cookie with session ID in it
* cookies are automatically managed by the browser
* sessions are saved in db, verified on every request
* the drawback of this is database access can be slow
#### token-based (stateless)
* once validation succeeds, server sends back a jwt token with user data like user ID, which contain a signature
* the token is a self-contained proof of authentication, server only verifies the signature
* the server no longer maintains sessions information
* this is after since the server do no perform db lookup at every request
* session management is more difficult since token is the only source of truth for sessions (wait for token to expire or invalidate all active sessions by use new signing keys for tokens)
#### Clerk
* token and session lifetime is decoupled: session is alive for days or weeks, but token are only valid for 1min to keep it short-lived (so when its compromised, it is likely already expired)
* after validation, clerk creates session db, but also a session token
* afterwards, the server only verifies the session token
## Kafka
* Definition: Kafka is a distributed system consisting of servers and clients that communicate via a high-performance TCP network protocol.
* Components:
	* kafka runs on a cluster of servers that are distributed. Servers that form the storage layer for kafka are called **brokers**
	* kafka clients allow one to read (**consumer**), write (**producer**) and process kafka events
* Mechanism:
	* Producers and consumers are responsible for writing and reading event data to Kafka, respectively.
	* Kafka events looks like consist of key, value, timestamp and optional metadata
	* Events are organized around topics; topics are **partitioned** on different Kafka brokers. Events with the same event key share the same partition.
* Why is Kafka so fast (https://www.youtube.com/watch?v=UNUz1-msbOM):
	* sequential IO: Kafka's primary data structure is append-only logs, data are written on hard drives sequentially.
	* read with zero copy between producer and consumer: this is saves several copy steps that exist in traditional network-disk data transfers.
* source: https://kafka.apache.org/documentation/#introduction
# Accessibility
## WCAG
* https://www.w3.org/TR/WCAG21
* Four principles:
	* Perceivable - Information and user interface components must be presentable to users in ways they can perceive. (alt text, contrast)
	* Operable - User interface components and navigation must be operable. (keyboard navigation)
	* Understandable - Information and the operation of user interface must be understandable. (language, input assistance)
	* Robust - Content must be robust enough that it can be interpreted reliably by a wide variety of user agents, including assistive technologies (assistive technology)
## Aria
* Aria helps with dynamic content and advanced user interface controls developed with HTML, JavaScript, and related technologies
* Add semantics to custom widgets to help user understand the widgets (help screen reader to announce it to user); does not alter behavior of the widget
* Aria authoring guide: https://www.w3.org/WAI/ARIA/apg/
# CMS
## WordPress
* Wordpress is a CMS
* headless CMS uses API to fetch data from databases, while WordPress is considered as a traditional CMS
* LAMP (Linux, Apache, MySQL, PHP) stack is one way of hosting a WordPress site, alternatively, one can use Nginx as the web server instead of apache
* Drupal is more customizable and complex than WordPress. It's similar to a web appliction to a blog
# Web Performance-Frontend Focused
* Fireship Vid: https://www.youtube.com/watch?v=0fONene3OIA
	* Core Web Vitals
		* LCP (Largest Contentful Paint): loading performance; LCP reports the render time of the largest image, text block, or video visible in the viewport, relative to when the user first navigated to the page
			* reduce resource load time (compress image, fewer fonts)
			* use CDN
			* blocking js (use server side rendering instead of client side)
		* FID (First Input Delay): interactivity; how long the web app takes to responde to interactions
			* reduce js by moving js to workers or lazy loading
		* CLS (Cumulative Layout Shift): measures every unexpected layout shift that occurs during the entire lifecycle of a page
			* set image size
* Google framework (https://web.dev/performance?hl=en)
* 14 Rules of Web Performance (https://youtu.be/HC1eVj5cQOo?si=6deqhNjAdyVth2TS)
# Performance
* N + 1 problem (https://stackoverflow.com/questions/97197/what-is-the-n1-selects-problem-in-orm-object-relational-mapping)
# Talk
* K8s design principles: probably Tim Hockin - Container Management at Google Scale - SCALE 13x
1. Kubernetes APIs are declarative rather than imperative. (extensible)
2. The Kubernetes control plane is transparent. There are no hidden internal APIs. (level triggered rather than event triggered, no single point of failure, immutable)
3. Meet the user where they are. (ease of migration)
4. Workload portability. (decouple distributed system program development and cluster implementation, k8s as os for applications)
*  Understand the Next Phase of Web Development - Steve Sanderson - NDC London 2024 