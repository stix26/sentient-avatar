#!/bin/bash

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed. Please install it first."
    echo "Visit: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "Please login to GitHub CLI first:"
    gh auth login
fi

# Repository settings
echo "Setting up repository settings..."
gh api --method PATCH /repos/stix26/sentient-avatar \
    -f has_issues=true \
    -f has_wiki=true \
    -f has_discussions=true \
    -f has_projects=true \
    -f allow_squash_merge=true \
    -f allow_merge_commit=true \
    -f allow_rebase_merge=true \
    -f delete_branch_on_merge=true

# Set repository topics
echo "Setting repository topics..."
gh api --method PUT /repos/stix26/sentient-avatar/topics \
    -f names='["ai","avatar","emotional-intelligence","fastapi","python","machine-learning","docker","microservices"]'

# Enable GitHub Pages
echo "Enabling GitHub Pages..."
gh api --method POST /repos/stix26/sentient-avatar/pages \
    -f source='{"branch":"gh-pages","path":"/"}'

# Set up branch protection
echo "Setting up branch protection..."
gh api --method PUT /repos/stix26/sentient-avatar/branches/main/protection \
    -f required_status_checks='{"strict":true,"contexts":["CI","Documentation"]}' \
    -f enforce_admins=true \
    -f required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
    -f restrictions=null

# Create issue labels
echo "Creating issue labels..."
gh api --method POST /repos/stix26/sentient-avatar/labels \
    -f name=bug \
    -f color=d73a4a \
    -f description="Something isn't working"

gh api --method POST /repos/stix26/sentient-avatar/labels \
    -f name=enhancement \
    -f color=a2eeef \
    -f description="New feature or request"

gh api --method POST /repos/stix26/sentient-avatar/labels \
    -f name=documentation \
    -f color=0075ca \
    -f description="Improvements or additions to documentation"

gh api --method POST /repos/stix26/sentient-avatar/labels \
    -f name="good first issue" \
    -f color=7057ff \
    -f description="Good for newcomers"

gh api --method POST /repos/stix26/sentient-avatar/labels \
    -f name="help wanted" \
    -f color=008672 \
    -f description="Extra attention is needed"

gh api --method POST /repos/stix26/sentient-avatar/labels \
    -f name=question \
    -f color=d876e3 \
    -f description="Further information is requested"

# Function to set a secret
set_secret() {
    local name=$1
    local value=$2
    echo "Setting secret: $name"
    gh secret set "$name" -b"$value" -R stix26/sentient-avatar
}

# Set up secrets
echo "Setting up repository secrets..."
echo "Please enter the following values:"

read -p "DockerHub Username: " dockerhub_username
read -sp "DockerHub Token: " dockerhub_token
echo
read -p "Deploy Host: " deploy_host
read -p "Deploy User: " deploy_user
read -sp "Deploy Key: " deploy_key
echo

# Set the secrets
set_secret "DOCKERHUB_USERNAME" "$dockerhub_username"
set_secret "DOCKERHUB_TOKEN" "$dockerhub_token"
set_secret "DEPLOY_HOST" "$deploy_host"
set_secret "DEPLOY_USER" "$deploy_user"
set_secret "DEPLOY_KEY" "$deploy_key"

echo "Repository setup complete!" 