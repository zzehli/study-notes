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
### Beej's Guide
### Sockets and IP addresses
* sockets: a way to speak to other programs using standard Unix file descriptors
* `SOCKE_STREAM` and `SOCK_DGRAM` for TCP and UDP, respectively
* OSI model
* IPv4 and IPv6: IPv4 is 4 bytes (32 bits), and IPv6 32 bytes (128 bits)
* subnets: first 1-3 # in a IPv4 represents the network, the reminder is for the host
    * add IP address to netmask to get the network number 
* port number: room numbers to street (aka IP) addresses
    * common port list: [IANA](https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml) 
* byte order: ordering of bytes in network transmission
    * wiki: Big-endianness is the dominant ordering in networking protocols, such as in the Internet protocol suite, where it is referred to as network order, transmitting the most significant byte first. Conversely, little-endianness is the dominant ordering for processor architectures (x86, most ARM implementations, base RISC-V implementations) and their associated memory
    * conversion: between big (network byte ordering) and little (host byte ordering) with functions for short or long numbers: `htons` `htonl` `ntohs` `ntohl`
* data structures:
    * socket descriptors: int
    * `addrinfo` socket addresses, linked list
    * part of `addrinfo` that holds actual addresses: `sockaddr`, `sockaddr_in`, `sockaddr_in16` and `sockaddr_storage`
        * `in_addr` or `in6_addr` are structs for addresses in these 
    * use `inet_pton` to convert addresses to `in_addr` or `in6_addr`, conversely, with `inet_ntop`
* private network done thru NAT (Network Address Translation) firewall
    * private network uses IP addresses like 10.x.x.x and 192.168.x.x

