import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
import logging

class GrafanaDashboard:
    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.logger = logging.getLogger(__name__)
    
    def create_dashboard(self, title: str, description: str) -> Dict:
        """Create a new dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "uid": None,
                "title": title,
                "tags": ["sentient-avatar"],
                "timezone": "browser",
                "schemaVersion": 30,
                "version": 1,
                "refresh": "10s",
                "description": description,
                "panels": []
            },
            "overwrite": True
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to create dashboard: {response.text}")
            raise Exception(f"Failed to create dashboard: {response.text}")
        
        return response.json()
    
    def add_panel(self, dashboard_uid: str, panel: Dict) -> Dict:
        """Add a panel to an existing dashboard."""
        dashboard = self.get_dashboard(dashboard_uid)
        dashboard["dashboard"]["panels"].append(panel)
        
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to add panel: {response.text}")
            raise Exception(f"Failed to add panel: {response.text}")
        
        return response.json()
    
    def get_dashboard(self, dashboard_uid: str) -> Dict:
        """Get dashboard by UID."""
        response = requests.get(
            f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}",
            headers=self.headers,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to get dashboard: {response.text}")
            raise Exception(f"Failed to get dashboard: {response.text}")
        
        return response.json()
    
    def create_system_dashboard(self) -> Dict:
        """Create system monitoring dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "uid": "system-monitoring",
                "title": "System Monitoring",
                "tags": ["sentient-avatar", "system"],
                "timezone": "browser",
                "schemaVersion": 30,
                "version": 1,
                "refresh": "10s",
                "description": "System resource monitoring dashboard",
                "panels": [
                    {
                        "title": "CPU Usage",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "rate(process_cpu_seconds_total[5m]) * 100",
                                "legendFormat": "CPU Usage"
                            }
                        ]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "process_resident_memory_bytes",
                                "legendFormat": "Memory Usage"
                            }
                        ]
                    },
                    {
                        "title": "Disk I/O",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "rate(node_disk_io_time_seconds_total[5m])",
                                "legendFormat": "Disk I/O"
                            }
                        ]
                    },
                    {
                        "title": "Network Traffic",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "rate(node_network_receive_bytes_total[5m])",
                                "legendFormat": "Network Receive"
                            },
                            {
                                "expr": "rate(node_network_transmit_bytes_total[5m])",
                                "legendFormat": "Network Transmit"
                            }
                        ]
                    }
                ]
            },
            "overwrite": True
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to create system dashboard: {response.text}")
            raise Exception(f"Failed to create system dashboard: {response.text}")
        
        return response.json()
    
    def create_service_dashboard(self) -> Dict:
        """Create service monitoring dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "uid": "service-monitoring",
                "title": "Service Monitoring",
                "tags": ["sentient-avatar", "services"],
                "timezone": "browser",
                "schemaVersion": 30,
                "version": 1,
                "refresh": "10s",
                "description": "Service health and performance monitoring",
                "panels": [
                    {
                        "title": "Service Health",
                        "type": "stat",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "service_health_status",
                                "legendFormat": "{{service}}"
                            }
                        ]
                    },
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{method}} {{path}}"
                            }
                        ]
                    },
                    {
                        "title": "Response Time",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
                                "legendFormat": "{{method}} {{path}}"
                            }
                        ]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "datasource": "Prometheus",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                                "legendFormat": "{{method}} {{path}}"
                            }
                        ]
                    }
                ]
            },
            "overwrite": True
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to create service dashboard: {response.text}")
            raise Exception(f"Failed to create service dashboard: {response.text}")
        
        return response.json()
    
    def create_alert_rule(self, name: str, condition: str, duration: str, severity: str) -> Dict:
        """Create a new alert rule."""
        alert_rule = {
            "name": name,
            "condition": condition,
            "duration": duration,
            "severity": severity,
            "notifications": []
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/v1/provisioning/alert-rules",
            headers=self.headers,
            json=alert_rule,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to create alert rule: {response.text}")
            raise Exception(f"Failed to create alert rule: {response.text}")
        
        return response.json()
    
    def create_notification_channel(self, name: str, type: str, settings: Dict) -> Dict:
        """Create a new notification channel."""
        channel = {
            "name": name,
            "type": type,
            "settings": settings,
            "isDefault": False
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/alert-notifications",
            headers=self.headers,
            json=channel,
            timeout=5
        )
        
        if response.status_code != 200:
            self.logger.error(f"Failed to create notification channel: {response.text}")
            raise Exception(f"Failed to create notification channel: {response.text}")
        
        return response.json()
    
    def setup_default_alerts(self) -> None:
        """Set up default alert rules."""
        # System alerts
        self.create_alert_rule(
            name="High CPU Usage",
            condition="avg(rate(process_cpu_seconds_total[5m])) > 0.8",
            duration="5m",
            severity="warning"
        )
        
        self.create_alert_rule(
            name="High Memory Usage",
            condition="process_resident_memory_bytes / node_memory_MemTotal_bytes > 0.9",
            duration="5m",
            severity="warning"
        )
        
        self.create_alert_rule(
            name="High Disk Usage",
            condition="node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.1",
            duration="5m",
            severity="warning"
        )
        
        # Service alerts
        self.create_alert_rule(
            name="Service Down",
            condition="service_health_status == 0",
            duration="1m",
            severity="critical"
        )
        
        self.create_alert_rule(
            name="High Error Rate",
            condition="rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) > 0.05",
            duration="5m",
            severity="warning"
        )
        
        self.create_alert_rule(
            name="High Response Time",
            condition="rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m]) > 1",
            duration="5m",
            severity="warning"
        )
    
    def setup_default_notifications(self) -> None:
        """Set up default notification channels."""
        # Email notifications
        self.create_notification_channel(
            name="Email Alerts",
            type="email",
            settings={
                "addresses": "admin@example.com",
                "singleEmail": True
            }
        )
        
        # Slack notifications
        self.create_notification_channel(
            name="Slack Alerts",
            type="slack",
            settings={
                "url": "https://hooks.slack.com/services/your-webhook-url",
                "recipient": "#alerts",
                "mentionChannel": "here"
            }
        )
        
        # PagerDuty notifications
        self.create_notification_channel(
            name="PagerDuty Alerts",
            type="pagerduty",
            settings={
                "integrationKey": "your-integration-key",
                "severity": "critical"
            }
        )
    
    def setup_monitoring(self) -> None:
        """Set up complete monitoring system."""
        try:
            # Create dashboards
            self.create_system_dashboard()
            self.create_service_dashboard()
            
            # Set up alerts
            self.setup_default_alerts()
            
            # Set up notifications
            self.setup_default_notifications()
            
            self.logger.info("Monitoring system setup completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to set up monitoring system: {str(e)}")
            raise 