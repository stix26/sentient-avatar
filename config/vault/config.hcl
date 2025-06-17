ui = true
disable_mlock = true

storage "raft" {
  path = "/vault/data"
  node_id = "vault_1"
  retry_join {
    leader_api_addr = "https://vault-1:8200"
  }
  retry_join {
    leader_api_addr = "https://vault-2:8200"
  }
  retry_join {
    leader_api_addr = "https://vault-3:8200"
  }
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = false
  tls_cert_file = "/vault/config/certs/vault.crt"
  tls_key_file = "/vault/config/certs/vault.key"
  tls_client_ca_file = "/vault/config/certs/ca.crt"
  tls_min_version = "tls12"
  tls_prefer_server_cipher_suites = true
  tls_cipher_suites = "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"
}

seal "transit" {
  address = "https://vault-transit:8200"
  token = "${TRANSIT_TOKEN}"
  disable_renewal = false
  key_name = "autounseal"
  mount_path = "transit/"
  tls_skip_verify = false
}

telemetry {
  prometheus_retention_time = "24h"
  disable_hostname = true
}

api_addr = "https://vault:8200"
cluster_addr = "https://vault:8201"

max_lease_ttl = "768h"
default_lease_ttl = "168h"

log_level = "info"
log_format = "json"
log_file = "/vault/logs/vault.log"

service_registration "consul" {
  address = "consul:8500"
} 