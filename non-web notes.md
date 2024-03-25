# Networking
# Programming language
## Go
* Simplicity is Complicated
# Tools
## GraphQL
* Think of it as a structured way of making REST calls: these queries are specific to your application, not to the database that holds your data.
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

### Talk
* K8s design principles
1. Kubernetes APIs are declarative rather than imperative. (extensible)
2. The Kubernetes control plane is transparent. There are no hidden internal APIs. (level triggered rather than event triggered, no single point of failure, immutable)
3. Meet the user where they are. (ease of migration)
4. Workload portability. (decouple distributed system program development and cluster implementation, k8s as os for applications)