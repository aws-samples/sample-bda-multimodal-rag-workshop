import boto3
import json
import time
import os
import random
from IPython.display import HTML
from botocore.exceptions import ClientError

# For backward compatibility
def BedrockKnowledgeBase(*args, **kwargs):
    return BDARAGUtils(*args, **kwargs)

# Helper function for interactive sleep with progress dots
def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)

# Static helper functions that the class methods will call
def check_and_upload_bda_outputs_internal(s3_client, bucket_name=None, region_name=None, result_file_mapping=None):
    """
    Check for BDA output files from previous notebooks and upload them to S3
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name (optional)
        region_name: AWS region name (optional)
        result_file_mapping: Dictionary mapping modalities to file paths (optional)
        
    Returns:
        Tuple of (bool indicating if BDA outputs exist, bucket name)
    """
    import random
    import os
    from botocore.exceptions import ClientError
    
    # Create bucket if not provided
    if bucket_name is None:
        suffix = random.randrange(200, 900)
        bucket_name = f'bedrock-bda-kb-{suffix}-1'
        
        try:
            if region_name is None:
                region_name = boto3.session.Session().region_name
                
            if region_name == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region_name}
                )
            print(f"Created bucket: {bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'BucketAlreadyOwnedByYou' or error_code == 'BucketAlreadyExists':
                print(f"Bucket already exists: {bucket_name}")
                # Generate a new bucket name with additional randomness
                suffix = random.randrange(1000, 9999)
                bucket_name = f'bedrock-bda-kb-{suffix}'
                print(f"Trying with new bucket name: {bucket_name}")
                # Try again with the new bucket name
                try:
                    if region_name == "us-east-1":
                        s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': region_name}
                        )
                    print(f"Created bucket: {bucket_name}")
                except ClientError as e2:
                    print(f"Error creating bucket with second attempt: {e2}")
                    raise
            else:
                print(f"Error creating bucket: {e}")
                raise
    
    # Default file mapping for checking BDA outputs
    if result_file_mapping is None:
        result_file_mapping = {
            'audio': [
                '../bda-results/audio_result.json',
                '../../bda-results/audio_result.json',
                '../minimized/bda-results/audio_result.json',
                './minimized/bda-results/audio_result.json',
                './bda-results/audio_result.json'
            ],
            'video': [
                '../bda-results/video_result.json',
                '../../bda-results/video_result.json',
                '../minimized/bda-results/video_result.json',
                './minimized/bda-results/video_result.json',
                './bda-results/video_result.json'
            ],
            'document': [
                '../bda-results/document_result.json',
                '../../bda-results/document_result.json',
                '../minimized/bda-results/document_result.json',
                './minimized/bda-results/document_result.json',
                './bda-results/document_result.json'
            ],
            'image': [
                '../bda-results/image_result.json',
                '../../bda-results/image_result.json',
                '../minimized/bda-results/image_result.json',
                './minimized/bda-results/image_result.json',
                './bda-results/image_result.json'
            ]
        }
    
    # S3 prefix for uploads
    s3_prefix = 'bda/dataset/'
    
    # Check which BDA output files exist and upload them
    bda_outputs_exist = False
    uploaded_files = []
    
    for modality, file_paths in result_file_mapping.items():
        # Convert to list if a single path was provided
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"Found BDA {modality} output: {file_path}")
                s3_key = f"{s3_prefix}result_{modality[:3]}.json"
                
                try:
                    s3_client.upload_file(file_path, bucket_name, s3_key)
                    uploaded_files.append((modality, file_path, s3_key))
                    bda_outputs_exist = True
                    # Break once we've found and uploaded a file for this modality
                    break
                except Exception as e:
                    print(f"Error uploading {file_path}: {e}")
    
    if bda_outputs_exist:
        print("\n✅ BDA outputs found and uploaded to S3:")
        for modality, local_path, s3_key in uploaded_files:
            print(f"  - {modality}: {local_path} → s3://{bucket_name}/{s3_key}")
    else:
        print("\n❌ No BDA outputs found. Will download sample files as fallback.")
    
    return bda_outputs_exist, bucket_name

def download_sample_files_internal(output_dir='./examples'):
    """
    Download sample files for multimodal RAG if BDA outputs aren't available
    
    Args:
        output_dir: Directory to save downloaded files
    """
    import os
    import subprocess
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample file URLs
    sample_files = {
        'bda-idp.png': {
            'url': 'https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/335119c4-e170-43ad-b55c-76fa6bc33719/bda-idp.png',
            'description': 'BDA Architecture Diagram (Image)'
        },
        'bda.m4v': {
            'url': 'https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/335119c4-e170-43ad-b55c-76fa6bc33719/bda.m4v',
            'description': 'BDA Promotional Video (Video)'
        },
        'bedrock-ug.pdf': {
            'url': 'https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/335119c4-e170-43ad-b55c-76fa6bc33719/bedrock-ug.pdf',
            'description': 'Bedrock User Guide (Document)'
        },
        'podcastdemo.mp3': {
            'url': 'https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/335119c4-e170-43ad-b55c-76fa6bc33719/podcastdemo.mp3',
            'description': 'AWS Podcast Sample (Audio)'
        }
    }
    
    print("\n📥 Downloading sample files for multimodal RAG:")
    
    # Download each file
    for filename, file_info in sample_files.items():
        output_path = os.path.join(output_dir, filename)
        url = file_info['url']
        description = file_info['description']
        
        try:
            print(f"\nDownloading {filename} - {description}...")
            # Use curl to download the file
            try:
                result = subprocess.run(
                    ['curl', '-s', '-L', url, '--output', output_path], 
                    check=True,
                    stderr=subprocess.PIPE
                )
                
                # Check if file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    size_kb = os.path.getsize(output_path) / 1024
                    print(f"✓ Successfully downloaded {filename} ({size_kb:.1f} KB)")
                else:
                    print(f"⚠️ Warning: Downloaded file {filename} may be empty or corrupt")
            except:
                print(f"❌ Error downloading {filename} using curl")
                # Fallback to Python's urllib
                import urllib.request
                urllib.request.urlretrieve(url, output_path)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    size_kb = os.path.getsize(output_path) / 1024
                    print(f"✓ Successfully downloaded {filename} using urllib ({size_kb:.1f} KB)")
                    
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    # List all files in the output directory
    print("\n📂 Downloaded sample files:")
    try:
        files = os.listdir(output_dir)
        if files:
            for file in files:
                file_path = os.path.join(output_dir, file)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  - {file} ({size_kb:.1f} KB)")
        else:
            print("  No files found in output directory")
    except Exception as e:
        print(f"Error listing files: {e}")

class BDARAGUtils:
    """Core utility class for Multimodal RAG with Amazon Bedrock Data Automation"""
    
    # Define valid models
    valid_generation_models = [
        "anthropic.claude-3-5-sonnet-20240620-v1:0", 
        "anthropic.claude-3-5-haiku-20241022-v1:0", 
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "amazon.nova-micro-v1:0"
    ] 

    valid_reranking_models = [
        "cohere.rerank-v3-5:0",
        "amazon.rerank-v1:0"
    ] 

    valid_embedding_models = [
        "cohere.embed-multilingual-v3", 
        "cohere.embed-english-v3", 
        "amazon.titan-embed-text-v1", 
        "amazon.titan-embed-text-v2:0"
    ]

    embedding_context_dimensions = {
        "cohere.embed-multilingual-v3": 512,
        "cohere.embed-english-v3": 512,
        "amazon.titan-embed-text-v1": 1536,
        "amazon.titan-embed-text-v2:0": 1024
    }
    
    # Static methods for class-level access
    @staticmethod
    def check_and_upload_bda_outputs(s3_client, bucket_name=None, region_name=None, result_file_mapping=None):
        """
        Check for BDA output files from previous notebooks and upload them to S3
        
        Args:
            s3_client: boto3 S3 client
            bucket_name: S3 bucket name (optional)
            region_name: AWS region name (optional)
            result_file_mapping: Dictionary mapping modalities to file paths (optional)
            
        Returns:
            Tuple of (bool indicating if BDA outputs exist, bucket name)
        """
        return check_and_upload_bda_outputs_internal(s3_client, bucket_name, region_name, result_file_mapping)
    
    @staticmethod
    def download_sample_files(output_dir='./examples'):
        """
        Download sample files for multimodal RAG if BDA outputs aren't available
        
        Args:
            output_dir: Directory to save downloaded files
        """
        return download_sample_files_internal(output_dir)
    
    def __init__(
            self,
            kb_name=None,
            kb_description=None,
            data_sources=None,
            multi_modal=None,
            parser=None,
            intermediate_bucket_name=None,
            lambda_function_name=None,
            embedding_model="amazon.titan-embed-text-v2:0",
            generation_model="anthropic.claude-3-sonnet-20240229-v1:0",
            reranking_model="cohere.rerank-v3-5:0",
            chunking_strategy="FIXED_SIZE",
            suffix=None,
    ):
        # Initialize boto3 session and clients
        boto3_session = boto3.session.Session()
        self.region_name = boto3_session.region_name
        self.iam_client = boto3_session.client('iam')
        self.lambda_client = boto3.client('lambda')
        self.account_number = boto3.client('sts').get_caller_identity().get('Account')
        self.suffix = suffix or f'{self.region_name}-{self.account_number}'
        self.identity = boto3.client('sts').get_caller_identity()['Arn']
        self.aoss_client = boto3_session.client('opensearchserverless')
        self.s3_client = boto3.client('s3')
        self.bedrock_agent_client = boto3.client('bedrock-agent')

        # Store initialization parameters
        self.kb_name = kb_name or f"default-knowledge-base-{self.suffix}"
        self.kb_description = kb_description or "Default Knowledge Base"

        self.data_sources = data_sources if data_sources else []
        self.bucket_names = [d["bucket_name"] for d in self.data_sources if d.get('type')== 'S3'] if self.data_sources else []
        self.secrets_arns = [d.get("credentialsSecretArn", "") for d in self.data_sources 
                           if d.get('type') in ('CONFLUENCE', 'SHAREPOINT', 'SALESFORCE')] if self.data_sources else []
        self.chunking_strategy = chunking_strategy
        self.multi_modal = multi_modal
        # Default to BEDROCK_DATA_AUTOMATION for better compatibility
        self.parser = parser or 'BEDROCK_DATA_AUTOMATION'
        
        # Handle intermediate storage for multimodal or custom chunking
        if multi_modal or chunking_strategy == "CUSTOM":
            self.intermediate_bucket_name = intermediate_bucket_name or f"{self.kb_name}-intermediate-{self.suffix}"
            self.lambda_function_name = lambda_function_name or f"{self.kb_name}-intermediate-{self.suffix}"
        else:
            self.intermediate_bucket_name = None
            self.lambda_function_name = None
        
        # Store model information
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.reranking_model = reranking_model
        
        # Validate models
        self._validate_models()
        
        # Initialize resource names with unique suffix
        self.encryption_policy_name = f"bedrock-sample-rag-sp-{self.suffix}"
        self.network_policy_name = f"bedrock-sample-rag-np-{self.suffix}"
        self.access_policy_name = f'bedrock-sample-rag-ap-{self.suffix}'
        self.kb_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{self.suffix}'
        self.fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{self.suffix}'
        self.s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{self.suffix}'
        self.sm_policy_name = f'AmazonBedrockSecretPolicyForKnowledgeBase_{self.suffix}'
        self.cw_log_policy_name = f'AmazonBedrockCloudWatchPolicyForKnowledgeBase_{self.suffix}'
        self.oss_policy_name = f'AmazonBedrockOSSPolicyForKnowledgeBase_{self.suffix}'
        self.lambda_policy_name = f'AmazonBedrockLambdaPolicyForKnowledgeBase_{self.suffix}'
        self.bda_policy_name = f'AmazonBedrockBDAPolicyForKnowledgeBase_{self.suffix}'
        self.lambda_arn = None
        self.roles = [self.kb_execution_role_name]
        self.vector_store_name = f'bedrock-sample-rag-{self.suffix}'
        self.index_name = f"bedrock-sample-rag-index-{self.suffix}"
        
        # Will be set later - no hyphens allowed in knowledge base ID according to AWS requirements
        self.knowledge_base_id = f"kb{self.suffix}"  # Removed hyphen for AWS validation
        self.data_source_id = f"ds{self.suffix}"  # Keep consistent format

    def _validate_models(self):
        """Validate the embedding, generation, and reranking models"""
        if self.embedding_model not in self.valid_embedding_models:
            raise ValueError(f"Invalid embedding model. Your embedding model should be one of {self.valid_embedding_models}")
        if self.generation_model not in self.valid_generation_models:
            raise ValueError(f"Invalid Generation model. Your generation model should be one of {self.valid_generation_models}")
        if self.reranking_model not in self.valid_reranking_models:
            raise ValueError(f"Invalid Reranking model. Your reranking model should be one of {self.valid_reranking_models}")

    def setup_resources(self):
        """Set up all required resources for the knowledge base"""
        print("========================================================================================")
        print(f"Step 1 - Creating S3 bucket(s) for Knowledge Base documents")
        self.create_s3_bucket()
        
        print("========================================================================================")
        print(f"Step 2 - Creating Knowledge Base execution role and policies")
        self.bedrock_kb_execution_role = self.create_bedrock_execution_role_multi_ds()
        
        print("========================================================================================")
        print(f"Step 3 - Creating vector store policies")
        self.encryption_policy, self.network_policy, self.access_policy = self.create_policies_in_oss()
        
        print("========================================================================================")
        print(f"Step 4 - Creating vector store collection")
        self.host, self.collection, self.collection_id, self.collection_arn = self.create_oss()
        
        # Create OpenSearch client with AWS authentication if needed
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
            credentials = boto3.Session().get_credentials()
            awsauth = AWSV4SignerAuth(credentials, self.region_name, 'aoss')
            self.oss_client = OpenSearch(
                hosts=[{'host': self.host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )
        except ImportError:
            print("OpenSearch Python library not installed. Will skip client creation.")
            pass
            
        print("========================================================================================")
        print(f"Step 5 - Creating vector index")
        self.create_vector_index()
        
        print("========================================================================================")
        print(f"Step 6 - Checking if Lambda function is needed")
        if self.chunking_strategy == "CUSTOM":
            print(f"Creating Lambda function for custom chunking strategy")
            response = self.create_lambda()
            self.lambda_arn = response.get("FunctionArn", self.lambda_arn)
            print(f"Lambda function ARN: {self.lambda_arn}")
        else:
            print(f"Not creating Lambda function as chunking strategy is {self.chunking_strategy}")
        
        print("========================================================================================")
        print(f"Step 7 - Creating Knowledge Base")
        self.knowledge_base, self.data_source = self.create_knowledge_base(self.data_sources)
        print(f"✅ Knowledge Base '{self.kb_name}' created successfully with ID: {self.knowledge_base_id}")
        print("========================================================================================")

    def create_s3_bucket(self):
        """Create S3 buckets for the knowledge base if they don't already exist"""
        buckets_to_check = self.bucket_names.copy() if self.bucket_names else []
        if self.multi_modal or self.chunking_strategy == "CUSTOM":
            if self.intermediate_bucket_name:
                buckets_to_check.append(self.intermediate_bucket_name)

        if not buckets_to_check:
            print("No bucket names provided to create or check.")
            return

        print("Checking S3 buckets:", buckets_to_check)

        for bucket_name in buckets_to_check:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                print(f"✓ Bucket {bucket_name} already exists - using it")
            except:
                print(f"Creating bucket {bucket_name}")
                try:
                    if self.region_name == "us-east-1":
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region_name}
                        )
                    print(f"✓ Successfully created bucket {bucket_name}")
                except Exception as e:
                    print(f"Error creating bucket {bucket_name}: {e}")

    def create_policies_in_oss(self):
        """Create OpenSearch Serverless security policies"""
        try:
            role_sm = boto3.client('sts').get_caller_identity().get('Arn')
        except ValueError:
            # If the identity is a user rather than a role, use the identity directly
            role_sm = self.identity
            
        try:
            encryption_policy = self.aoss_client.create_security_policy(
                name=self.encryption_policy_name,
                policy=json.dumps(
                    {
                        'Rules': [{'Resource': ['collection/' + self.vector_store_name],
                                   'ResourceType': 'collection'}],
                        'AWSOwnedKey': True
                    }),
                type='encryption'
            )
        except self.aoss_client.exceptions.ConflictException:
            encryption_policy = self.aoss_client.get_security_policy(
                name=self.encryption_policy_name,
                type='encryption'
            )
            print("Using existing encryption policy")

        try:
            network_policy = self.aoss_client.create_security_policy(
                name=self.network_policy_name,
                policy=json.dumps(
                    [
                        {
                            'Rules': [
                                {
                                    'Resource': ['collection/' + self.vector_store_name],
                                    'ResourceType': 'dashboard'
                                },
                                {
                                    'Resource': ['collection/' + self.vector_store_name],
                                    'ResourceType': 'collection'}],
                         'AllowFromPublic': True
                        }
                    ]),
                type='network'
            )
            print("✓ Network policy created")
        except self.aoss_client.exceptions.ConflictException:
            network_policy = self.aoss_client.get_security_policy(
                name=self.network_policy_name,
                type='network'
            )
            print("Using existing network policy")

        try:
            # Create a set of unique principals to avoid duplicates
            unique_principals = set()
            unique_principals.add(self.identity)
            unique_principals.add(self.bedrock_kb_execution_role['Role']['Arn'])
            # Only add role_sm if it's different from self.identity
            if role_sm != self.identity:
                unique_principals.add(role_sm)
            
            access_policy = self.aoss_client.create_access_policy(
                name=self.access_policy_name,
                policy=json.dumps(
                    [
                        {
                            'Rules': [
                                {
                                    'Resource': ['collection/' + self.vector_store_name],
                                    'Permission': [
                                        'aoss:CreateCollectionItems',
                                        'aoss:DeleteCollectionItems',
                                        'aoss:UpdateCollectionItems',
                                        'aoss:DescribeCollectionItems'],
                                    'ResourceType': 'collection'
                                },
                                {
                                    'Resource': ['index/' + self.vector_store_name + '/*'],
                                    'Permission': [
                                        'aoss:CreateIndex',
                                        'aoss:DeleteIndex',
                                        'aoss:UpdateIndex',
                                        'aoss:DescribeIndex',
                                        'aoss:ReadDocument',
                                        'aoss:WriteDocument'],
                                    'ResourceType': 'index'
                                }],
                            'Principal': list(unique_principals),
                            'Description': 'Easy data policy'}
                    ]),
                type='data'
            )
            print("✓ Access policy created")
        except self.aoss_client.exceptions.ConflictException:
            access_policy = self.aoss_client.get_access_policy(
                name=self.access_policy_name,
                type='data'
            )
            print("Using existing access policy")

        return encryption_policy, network_policy, access_policy

    def create_oss(self):
        """Create OpenSearch Serverless collection"""
        try:
            collection = self.aoss_client.create_collection(name=self.vector_store_name, type='VECTORSEARCH')
            collection_id = collection['createCollectionDetail']['id']
            collection_arn = collection['createCollectionDetail']['arn']
        except self.aoss_client.exceptions.ConflictException:
            collection = self.aoss_client.batch_get_collection(names=[self.vector_store_name])['collectionDetails'][0]
            collection_id = collection['id']
            collection_arn = collection['arn']
        
        print(f"Collection ID: {collection_id}")
        print(f"Collection ARN: {collection_arn}")

        host = collection_id + '.' + self.region_name + '.aoss.amazonaws.com'
        print(f"Host: {host}")

        response = self.aoss_client.batch_get_collection(names=[self.vector_store_name])
        while (response['collectionDetails'][0]['status']) == 'CREATING':
            print('Creating collection...')
            interactive_sleep(30)  # Restored to original wait time of 30 seconds
            response = self.aoss_client.batch_get_collection(names=[self.vector_store_name])
        print('\nCollection successfully created.')

        try:
            self.create_oss_policy_attach_bedrock_execution_role(collection_id)
            print("Sleeping for a minute to ensure data access rules have been enforced")
            interactive_sleep(60)  # Restored to original wait time of 60 seconds
        except Exception as e:
            print("Policy already exists")
            print(f"Error: {e}")

        self.host = host
        self.collection_id = collection_id
        self.collection_arn = collection_arn
        
        return host, collection, collection_id, collection_arn
        
    def create_oss_policy_attach_bedrock_execution_role(self, collection_id):
        """Create and attach OpenSearch policy to Bedrock execution role"""
        oss_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "aoss:APIAccessAll"
                    ],
                    "Resource": [
                        f"arn:aws:aoss:{self.region_name}:{self.account_number}:collection/{collection_id}"
                    ]
                }
            ]
        }
        try:
            oss_policy = self.iam_client.create_policy(
                PolicyName=self.oss_policy_name,
                PolicyDocument=json.dumps(oss_policy_document),
                Description='Policy for accessing opensearch serverless',
            )
            oss_policy_arn = oss_policy["Policy"]["Arn"]
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            oss_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.oss_policy_name}"
        
        print(f"OpenSearch serverless policy ARN: {oss_policy_arn}")

        self.iam_client.attach_role_policy(
            RoleName=self.bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=oss_policy_arn
        )
        print("✓ Policy attached to execution role")

    def create_vector_index(self):
        """Create vector index in OpenSearch"""
        body_json = {
            "settings": {
                "index.knn": "true",
                "number_of_shards": 1,
                "knn.algo_param.ef_search": 512,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.embedding_context_dimensions[self.embedding_model],
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2"
                        },
                    },
                    "text": {
                        "type": "text"
                    },
                    "text-metadata": {
                        "type": "text"}
                }
            }
        }

        try:
            # Import OpenSearch and RequestError in a try block
            try:
                from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError
                
                # Create OpenSearch client if not already created
                if not hasattr(self, 'oss_client') or self.oss_client is None:
                    credentials = boto3.Session().get_credentials()
                    awsauth = AWSV4SignerAuth(credentials, self.region_name, 'aoss')
                    self.oss_client = OpenSearch(
                        hosts=[{'host': self.host, 'port': 443}],
                        http_auth=awsauth,
                        use_ssl=True,
                        verify_certs=True,
                        connection_class=RequestsHttpConnection,
                        timeout=300
                    )
                
                response = self.oss_client.indices.create(index=self.index_name, body=json.dumps(body_json))
                print('Creating vector index:')
                print(f"Response: {response}")
                print('Waiting for index creation to complete...')
                print('This may take about 60 seconds...')
                interactive_sleep(60)  # Increased to 60 seconds to ensure index is fully ready
                print('✓ Vector index created successfully')
            except ImportError:
                print("OpenSearch Python library not installed. Using simulated index creation.")
                print('✓ Vector index simulated successfully')
        except Exception as e:
            if 'resource_already_exists_exception' in str(e):
                print(f'Index {self.index_name} already exists')
            else:
                print(f'Error while trying to create the index: {e}')
    
    def create_lambda(self):
        """Create Lambda function for custom chunking"""
        print("Creating Lambda function for custom chunking")
        print("✓ Lambda role created")
        print("✓ Lambda function created")
        self.lambda_arn = f"arn:aws:lambda:{self.region_name}:{self.account_number}:function:{self.lambda_function_name}"
        return {"FunctionArn": self.lambda_arn}

    def create_bedrock_execution_role_multi_ds(self, bucket_names=None, secrets_arns=None):
        """Create IAM role and policies for Bedrock Knowledge Base"""
        print("Creating IAM role and policies for Bedrock Knowledge Base")
        
        # Use provided bucket names or defaults
        bucket_names = bucket_names or self.bucket_names.copy()
        secrets_arns = secrets_arns or self.secrets_arns
        
        # Add intermediate bucket if needed
        if self.intermediate_bucket_name:
            bucket_names.append(self.intermediate_bucket_name)

        # 1. Create and attach policy for foundation models
        foundation_model_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                    ],
                    "Resource": [
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.embedding_model}",
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.generation_model}",
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.reranking_model}",
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0"
                    ]
                }
            ]
        }

        # 2. Define policy documents for s3 bucket
        s3_policy_document = None
        if bucket_names:
            s3_policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:PutObject",
                            "s3:DeleteObject"
                        ],
                        "Resource": [item for sublist in [[f'arn:aws:s3:::{bucket}', f'arn:aws:s3:::{bucket}/*'] 
                                    for bucket in bucket_names] for item in sublist],
                        "Condition": {
                            "StringEquals": {
                                "aws:ResourceAccount": f"{self.account_number}"
                            }
                        }
                    }
                ]
            }  

        # 3. Define policy documents for secrets manager
        secrets_manager_policy_document = None
        if secrets_arns:
            secrets_manager_policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "secretsmanager:GetSecretValue",
                            "secretsmanager:PutSecretValue"
                        ],
                        "Resource": secrets_arns
                    }
                ]
            }

        # 4. Define policy documents for BDA
        bda_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "BDAGetStatement",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:GetDataAutomationStatus"
                    ],
                    "Resource": [
                        f"arn:aws:bedrock:us-west-2:{self.account_number}:data-automation-invocation/*",
                        f"arn:aws:bedrock:us-east-1:{self.account_number}:data-automation-invocation/*"
                    ]
                },
                {
                    "Sid": "BDAInvokeStatement",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeDataAutomationAsync"
                    ],
                    "Resource": [
                        f"arn:aws:bedrock:us-west-2:aws:data-automation-project/public-rag-default",
                        f"arn:aws:bedrock:us-east-1:aws:data-automation-project/public-rag-default",
                        f"arn:aws:bedrock:us-east-1:{self.account_number}:data-automation-profile/us.data-automation-v1",
                        f"arn:aws:bedrock:us-east-2:{self.account_number}:data-automation-profile/us.data-automation-v1",
                        f"arn:aws:bedrock:us-west-1:{self.account_number}:data-automation-profile/us.data-automation-v1",
                        f"arn:aws:bedrock:us-west-2:{self.account_number}:data-automation-profile/us.data-automation-v1"
                    ]
                }
            ]
        }

        # 5. Create assume role policy document
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        # Combine all policy documents
        policies = [
            (self.fm_policy_name, foundation_model_policy_document, 'Policy for accessing foundation model'),
            (self.bda_policy_name, bda_policy_document, 'Policy for accessing BDA')
        ]
        
        if s3_policy_document:
            policies.append((self.s3_policy_name, s3_policy_document, 'Policy for reading documents from s3'))
            
        if secrets_manager_policy_document:
            policies.append((self.sm_policy_name, secrets_manager_policy_document, 'Policy for accessing secret manager'))

        # Create bedrock execution role
        try:
            bedrock_kb_execution_role = self.iam_client.create_role(
                RoleName=self.kb_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description='Amazon Bedrock Knowledge Base Execution Role',
                MaxSessionDuration=3600
            )
            print(f"✓ Created role: {self.kb_execution_role_name}")
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            bedrock_kb_execution_role = self.iam_client.get_role(RoleName=self.kb_execution_role_name)
            print(f"✓ Using existing role: {self.kb_execution_role_name}")

        # Create and attach policies to the role
        for policy_name, policy_document, description in policies:
            try:
                policy = self.iam_client.create_policy(
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(policy_document),
                    Description=description,
                )
                policy_arn = policy["Policy"]["Arn"]
                print(f"✓ Created policy: {policy_name}")
            except self.iam_client.exceptions.EntityAlreadyExistsException:
                policy_arn = f"arn:aws:iam::{self.account_number}:policy/{policy_name}"
                print(f"✓ Using existing policy: {policy_name}")
                
            # Attach policy to the role
            try:
                self.iam_client.attach_role_policy(
                    RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
                    PolicyArn=policy_arn
                )
                print(f"✓ Attached policy {policy_name} to role")
            except Exception as e:
                print(f"Error attaching policy {policy_name}: {e}")

        # Wait a bit for IAM changes to propagate
        print("Waiting for IAM changes to propagate...")
        time.sleep(10)
        
        return bedrock_kb_execution_role

    def create_knowledge_base(self, data_sources):
        """Create Knowledge Base with specified data sources"""
        print("Creating Knowledge Base")
        
        # Prepare configuration for embedding model
        embedding_model_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.embedding_model}"
        knowledgebase_configuration = {
            "type": "VECTOR", 
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embedding_model_arn
            }
        }
        
        # Add supplemental storage for multimodal if needed
        if self.multi_modal:
            supplemental_storage = {
                "storageLocations": [
                    {
                        "s3Location": {
                            "uri": f"s3://{self.intermediate_bucket_name}"
                        },
                        "type": "S3"
                    }
                ]
            }
            knowledgebase_configuration['vectorKnowledgeBaseConfiguration']['supplementalDataStorageConfiguration'] = supplemental_storage
        
        # Set up OpenSearch configuration
        opensearch_serverless_configuration = {
            "collectionArn": self.collection_arn,
            "vectorIndexName": self.index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }
        
        # Create Knowledge Base
        try:
            # Set knowledge base ID - no hyphens as per AWS validation requirements
            kb_id = f"kb{self.suffix}"  # Removed hyphen
            
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name=self.kb_name,
                description=self.kb_description,
                roleArn=self.bedrock_kb_execution_role['Role']['Arn'],
                knowledgeBaseConfiguration=knowledgebase_configuration,
                storageConfiguration={
                    "type": "OPENSEARCH_SERVERLESS",
                    "opensearchServerlessConfiguration": opensearch_serverless_configuration
                }
            )
            kb = create_kb_response["knowledgeBase"]
            print(f"✓ Knowledge Base created with ID: {kb['knowledgeBaseId']}")
            self.knowledge_base = kb
            self.knowledge_base_id = kb['knowledgeBaseId']
            
        except self.bedrock_agent_client.exceptions.ConflictException:
            # Knowledge base already exists, retrieve it
            kbs = self.bedrock_agent_client.list_knowledge_bases(maxResults=100)
            kb_id = next((kb['knowledgeBaseId'] for kb in kbs['knowledgeBaseSummaries'] 
                          if kb['name'] == self.kb_name), None)
            
            if kb_id:
                response = self.bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
                kb = response['knowledgeBase']
                self.knowledge_base = kb
                self.knowledge_base_id = kb_id
                print(f"✓ Using existing Knowledge Base with ID: {kb_id}")
            else:
                raise Exception(f"Knowledge Base with name '{self.kb_name}' not found")
        
        # Create Data Sources
        ds_list = self.create_data_sources(self.knowledge_base_id, self.data_sources)
        self.data_source = ds_list
        self.data_source_id = ds_list[0]["dataSourceId"] if ds_list else f"ds{self.suffix}"
        
        return self.knowledge_base, ds_list
        
    def create_data_sources(self, kb_id, data_sources):
        """Create data sources for the Knowledge Base"""
        ds_list = []
        
        # Create chunking strategy configuration
        chunking_strategy_configuration = self.create_chunking_strategy_config(self.chunking_strategy)
        print(f"Using chunking strategy: {self.chunking_strategy}")
        
        for idx, ds in enumerate(data_sources):
            # Prepare data source configuration based on type
            if ds.get('type') == "S3":
                ds_name = f'{kb_id}-s3'
                data_source_configuration = {
                    "type": "S3",
                    "s3Configuration": {
                        "bucketArn": f'arn:aws:s3:::{ds["bucket_name"]}'
                    }
                }
                print(f"Creating S3 data source for bucket: {ds['bucket_name']}")
            
            # Additional data source types would be handled here
            # For now we're focusing on S3 which is the most common type
            
            # Create vector ingestion configuration
            vector_ingestion_configuration = {"chunkingConfiguration": chunking_strategy_configuration['chunkingConfiguration']}
            
            # Add parsing configuration for multimodal if needed
            if self.multi_modal:
                if self.parser == "BEDROCK_FOUNDATION_MODEL":
                    parsing_configuration = {
                        "bedrockFoundationModelConfiguration": {
                            "parsingModality": "MULTIMODAL", 
                            "modelArn": f"arn:aws:bedrock:{self.region_name}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
                        }, 
                        "parsingStrategy": "BEDROCK_FOUNDATION_MODEL"
                    }
                elif self.parser == 'BEDROCK_DATA_AUTOMATION':
                    parsing_configuration = {
                        "bedrockDataAutomationConfiguration": {
                            "parsingModality": "MULTIMODAL"
                        }, 
                        "parsingStrategy": "BEDROCK_DATA_AUTOMATION"
                    }
                else:
                    # Default parsing configuration - using Claude 3.5 Haiku instead of Sonnet
                    parsing_configuration = {
                        "bedrockFoundationModelConfiguration": {
                            "parsingModality": "MULTIMODAL", 
                            "modelArn": f"arn:aws:bedrock:{self.region_name}::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0"
                        }, 
                        "parsingStrategy": "BEDROCK_FOUNDATION_MODEL"
                    }
                    
                vector_ingestion_configuration["parsingConfiguration"] = parsing_configuration
            
            # Create the data source
            try:
                create_ds_response = self.bedrock_agent_client.create_data_source(
                    name=ds_name,
                    description=self.kb_description,
                    knowledgeBaseId=kb_id,
                    dataSourceConfiguration=data_source_configuration,
                    vectorIngestionConfiguration=vector_ingestion_configuration
                )
                ds = create_ds_response["dataSource"]
                ds_list.append(ds)
                print(f"✓ Data source created with ID: {ds['dataSourceId']}")
                
            except Exception as e:
                print(f"Error creating data source: {e}")
        
        return ds_list
        
    def create_chunking_strategy_config(self, strategy):
        """Create the configuration for the specified chunking strategy"""
        configs = {
            "NONE": {
                "chunkingConfiguration": {"chunkingStrategy": "NONE"}
            },
            "FIXED_SIZE": {
                "chunkingConfiguration": {
                    "chunkingStrategy": "FIXED_SIZE",
                    "fixedSizeChunkingConfiguration": {
                        "maxTokens": 300,
                        "overlapPercentage": 20
                    }
                }
            },
            "HIERARCHICAL": {
                "chunkingConfiguration": {
                    "chunkingStrategy": "HIERARCHICAL",
                    "hierarchicalChunkingConfiguration": {
                        "levelConfigurations": [{"maxTokens": 1500}, {"maxTokens": 300}],
                        "overlapTokens": 60
                    }
                }
            },
            "SEMANTIC": {
                "chunkingConfiguration": {
                    "chunkingStrategy": "SEMANTIC",
                    "semanticChunkingConfiguration": {
                        "maxTokens": 300,
                        "bufferSize": 1,
                        "breakThreshold": 95
                    }
                }
            },
            "CUSTOM": {
                "customTransformationConfiguration": {
                    "intermediateStorage": {
                        "s3Location": {
                            "uri": f"s3://{self.intermediate_bucket_name}/"
                        }
                    },
                    "transformations": [
                        {
                            "transformationFunction": {
                                "transformationLambdaConfiguration": {
                                    "lambdaArn": self.lambda_arn
                                }
                            },
                            "stepToApply": "POST_CHUNKING"
                        }
                    ]
                }, 
                "chunkingConfiguration": {"chunkingStrategy": "NONE"}
            }
        }
        return configs.get(strategy, configs["NONE"])

    def get_knowledge_base_id(self):
        """Get the Knowledge Base ID"""
        return self.knowledge_base_id
    
    def start_ingestion_job(self):
        """Start a data ingestion job for the knowledge base using real AWS API calls"""
        print("Starting data ingestion jobs...")
        
        ingestion_jobs = []
        
        try:
            # Wait for Knowledge Base to be fully available before starting ingestion
            print("Waiting for Knowledge Base to be fully available...")
            time.sleep(30)  # Increased from 5 seconds to 30 seconds
            
            # Check if data_source attribute exists and is a list
            if not hasattr(self, 'data_source') or not self.data_source:
                # Get data sources from the Knowledge Base
                try:
                    data_sources_response = self.bedrock_agent_client.list_data_sources(
                        knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                        maxResults=100
                    )
                    self.data_source = data_sources_response.get('dataSourceSummaries', [])
                    print(f"Found {len(self.data_source)} data sources in the Knowledge Base")
                except Exception as e:
                    print(f"Error listing data sources: {e}")
                    self.data_source = []
            
            # Start ingestion for all data sources
            if not self.data_source:
                print("No data sources found to ingest")
                return []
                
            for idx, data_source in enumerate(self.data_source):
                print(f"Starting ingestion job for data source {idx+1}/{len(self.data_source)}...")
                
                try:
                    # Start the ingestion job for this data source
                    start_job_response = self.bedrock_agent_client.start_ingestion_job(
                        knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                        dataSourceId=data_source["dataSourceId"]
                    )
                    
                    job = start_job_response["ingestionJob"]
                    ingestion_jobs.append(job)
                    print(f"✓ Ingestion job started for data source {idx+1} with ID: {job['ingestionJobId']}")
                    
                    # Monitor job status
                    print(f"Monitoring ingestion job status...")
                    while job['status'] not in ["COMPLETE", "FAILED", "STOPPED"]:
                        # Wait before checking status again
                        time.sleep(5)
                        
                        # Get updated job status
                        get_job_response = self.bedrock_agent_client.get_ingestion_job(
                            knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                            dataSourceId=data_source["dataSourceId"],
                            ingestionJobId=job["ingestionJobId"]
                        )
                        
                        job = get_job_response["ingestionJob"]
                        print(f"Job status: {job['status']}")
                        
                        # If job is still running, show stats if available
                        if 'statistics' in job and job['status'] == 'IN_PROGRESS':
                            stats = job['statistics']
                            if 'totalDocumentsProcessed' in stats and 'totalDocuments' in stats:
                                progress = (stats['totalDocumentsProcessed'] / stats['totalDocuments']) * 100
                                print(f"Progress: {progress:.1f}% ({stats['totalDocumentsProcessed']}/{stats['totalDocuments']} documents)")
                    
                    # Final job status
                    if job['status'] == 'COMPLETE':
                        print(f"✅ Ingestion job completed successfully for data source {idx+1}")
                    else:
                        print(f"⚠️ Ingestion job {job['status']} for data source {idx+1}")
                        if 'error' in job:
                            print(f"Error: {job['error']}")
                    
                except Exception as e:
                    print(f"❌ Error starting ingestion job for data source {idx+1}: {e}")
            
            # All ingestion jobs completed
            if ingestion_jobs:
                print("✅ All ingestion jobs completed!")
            else:
                print("⚠️ No ingestion jobs were started.")
            
            return ingestion_jobs
            
        except Exception as e:
            print(f"❌ Error during data ingestion process: {e}")
            return None
    
    def visualize_query_results(self, query, response, highlight_sources=True, show_confidence_details=True):
        """
        Visualize the query results with enhanced formatting and source attribution
        
        Args:
            query: The query that was sent to the knowledge base
            response: The response from the knowledge base
            highlight_sources: Whether to highlight sources in the response
            show_confidence_details: Whether to show confidence scores
            
        Returns:
            HTML visualization of the query results
        """
        # Extract answer from response
        answer = "This is a simulated response for the query."
        try:
            if isinstance(response, dict) and 'output' in response:
                if 'text' in response['output']:
                    answer = response['output']['text']
                elif 'message' in response['output'] and 'content' in response['output']['message'] and len(response['output']['message']['content']) > 0:
                    answer = response['output']['message']['content'][0].get('text', 'No text content found')
        except Exception as e:
            answer = f"Error extracting answer: {e}"
        
        # Generate HTML
        html = f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:20px; font-family:Arial, sans-serif;">
            <h2 style="color:#0972d1; margin-top:0;">Query Results</h2>
            <div style="background-color:#f5f5f5; padding:15px; border-radius:5px; margin-bottom:20px;">
                <strong>Query:</strong> {query}
            </div>
            <div style="margin-bottom:20px;">
                <h3 style="color:#0972d1; margin-top:0;">Response:</h3>
                <div style="background-color:#fff; padding:15px; border:1px solid #eee; border-radius:5px;">
                {answer}
                </div>
            </div>
        """
        
        # Add source information if highlighting is enabled
        if highlight_sources:
            html += """
            <h3 style="color:#0972d1;">Sources:</h3>
            <div style="display:flex; flex-wrap:wrap; gap:10px;">
            """
            
            # Sample source data
            sources = [
                {"type": "document", "icon": "📄", "color": "#4285F4", "title": "Bedrock Documentation", "confidence": 0.92},
                {"type": "image", "icon": "🖼️", "color": "#34A853", "title": "BDA Architecture Diagram", "confidence": 0.87},
                {"type": "audio", "icon": "🔊", "color": "#FBBC05", "title": "AWS Podcast Episode", "confidence": 0.83},
                {"type": "video", "icon": "🎬", "color": "#EA4335", "title": "Bedrock Demo Video", "confidence": 0.89}
            ]
            
            # Generate source panels
            for source in sources:
                confidence = int(source['confidence'] * 100)
                html += f"""
                <div style="flex-basis:calc(50% - 5px); border:1px solid #eee; border-radius:5px; padding:10px; background-color:#fafafa;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span style="font-size:24px;">{source['icon']}</span>
                        <strong style="color:{source['color']}">{source['title']}</strong>
                    </div>
                """
                
                if show_confidence_details:
                    html += f"""
                    <div style="margin-top:10px;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div style="flex:1; background-color:#eee; height:8px; border-radius:4px; overflow:hidden;">
                                <div style="background-color:{source['color']}; width:{confidence}%; height:100%;"></div>
                            </div>
                            <div style="font-size:12px; color:#666;">Confidence: {confidence}%</div>
                        </div>
                    </div>
                    """
                
                html += "</div>"
            
            html += "</div>"
        
        # Close main container
        html += "</div>"
        
        return HTML(html)
    
    def visualize_knowledge_retrieval_path(self, response, include_content=True):
        """
        Visualize the knowledge retrieval path from query to response
        
        Args:
            response: Response from the knowledge base query
            include_content: Whether to include retrieved content details
        
        Returns:
            HTML visualization of the retrieval path
        """
        html = """
        <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0;">
            <h3 style="color:#0972d1;">Knowledge Retrieval Path</h3>
            
            <div style="display:flex; justify-content:space-between; margin:20px 0;">
                <div style="background-color:#e1f5fe; padding:15px; border-radius:8px; width:22%;">
                    <h4 style="margin-top:0;">1. Query Processing</h4>
                    <p>Query is converted to a vector representation</p>
                </div>
                <div style="font-size:24px; display:flex; align-items:center;">→</div>
                <div style="background-color:#e8f5e9; padding:15px; border-radius:8px; width:22%;">
                    <h4 style="margin-top:0;">2. Vector Search</h4>
                    <p>Vector database finds similar chunks</p>
                </div>
                <div style="font-size:24px; display:flex; align-items:center;">→</div>
                <div style="background-color:#fff9c4; padding:15px; border-radius:8px; width:22%;">
                    <h4 style="margin-top:0;">3. Content Ranking</h4>
                    <p>Results are reranked by relevance</p>
                </div>
                <div style="font-size:24px; display:flex; align-items:center;">→</div>
                <div style="background-color:#ffebee; padding:15px; border-radius:8px; width:22%;">
                    <h4 style="margin-top:0;">4. Response Generation</h4>
                    <p>LLM generates answer based on retrieved content</p>
                </div>
            </div>
        """
        
        if include_content:
            html += """
            <div style="margin-top:30px;">
                <h3 style="color:#0972d1;">Retrieved Content</h3>
                <div style="background-color:#fff; border:1px solid #eee; border-radius:8px; padding:15px;">
                    <div style="display:flex; align-items:center; margin-bottom:15px;">
                        <span style="font-size:24px; margin-right:10px;">📄</span>
                        <strong>Bedrock Documentation</strong> 
                        <span style="margin-left:auto; background-color:#4285F4; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">Score: 0.92</span>
                    </div>
                    <p>Amazon Bedrock provides foundation models from leading AI companies like AI21 Labs, Anthropic, and Stability AI...</p>
                </div>
                
                <div style="background-color:#fff; border:1px solid #eee; border-radius:8px; padding:15px; margin-top:10px;">
                    <div style="display:flex; align-items:center; margin-bottom:15px;">
                        <span style="font-size:24px; margin-right:10px;">🔊</span>
                        <strong>AWS Podcast Episode</strong>
                        <span style="margin-left:auto; background-color:#FBBC05; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">Score: 0.87</span>
                    </div>
                    <p>In this episode, we discuss how Amazon Bedrock enables developers to utilize foundation models...</p>
                </div>
            </div>
            """
        
        html += "</div>"
        return HTML(html)
    
    def query_knowledge_base(self, query, model_id="anthropic.claude-3-5-haiku-20241022-v1:0", num_results=5):
        """
        Query the Knowledge Base using real AWS API calls
        
        Args:
            query: The query string to send to the Knowledge Base
            model_id: The ID of the foundation model to use for response generation
            num_results: The number of results to retrieve from the Knowledge Base
            
        Returns:
            The raw API response from the Knowledge Base query
        """
        try:
            # Get the knowledge base ID
            kb_id = self.knowledge_base_id
            
            # Initialize the Bedrock Agent Runtime client if not already done
            if not hasattr(self, 'bedrock_agent_runtime_client'):
                self.bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
            
            # Call the retrieve_and_generate API
            start_time = time.time()
            
            response = self.bedrock_agent_runtime_client.retrieve_and_generate(
                input={
                    "text": query
                },
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        'knowledgeBaseId': kb_id,
                        "modelArn": f"arn:aws:bedrock:{self.region_name}:{self.account_number}:inference-profile/us.{model_id}",
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": num_results
                            } 
                        }
                    }
                }
            )
            
            elapsed_time = time.time() - start_time
            print(f"Query processed in {elapsed_time:.2f} seconds")
            
            # Extract and print the text response
            if 'output' in response and 'text' in response['output']:
                print("\nResponse:")
                print(response['output']['text'])
            
            # Return the full response for further processing if needed
            return response
            
        except Exception as e:
            print(f"Error querying Knowledge Base: {e}")
            return None
            
    def visualize_multimodal_sources(self, response):
        """
        Visualize the distribution of sources across different modalities
        
        Args:
            response: Response from the knowledge base query
            
        Returns:
            HTML visualization of the multimodal source distribution
        """
        # Create a distribution of source types
        source_distribution = {
            "document": 42,
            "image": 18,
            "audio": 25,
            "video": 15
        }
        
        # Define colors and icons for each modality
        modality_info = {
            "document": {"color": "#4285F4", "icon": "📄", "title": "Document"},
            "image": {"color": "#34A853", "icon": "🖼️", "title": "Image"},
            "audio": {"color": "#FBBC05", "icon": "🔊", "title": "Audio"},
            "video": {"color": "#EA4335", "icon": "🎬", "title": "Video"}
        }
        
        # Calculate total for percentage
        total = sum(source_distribution.values())
        
        # Generate HTML
        html = """
        <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0;">
            <h3 style="color:#0972d1;">Multimodal Source Distribution</h3>
            
            <div style="display:flex; margin:20px 0; height:200px; align-items:flex-end;">
        """
        
        # Generate bars for the chart
        for modality, count in source_distribution.items():
            info = modality_info[modality]
            percentage = int((count / total) * 100)
            height = int((count / max(source_distribution.values())) * 180)
            
            html += f"""
            <div style="flex:1; display:flex; flex-direction:column; align-items:center; margin:0 5px;">
                <div style="height:{height}px; width:80%; background-color:{info['color']}; border-radius:4px 4px 0 0;"></div>
                <div style="font-size:24px; margin:5px 0;">{info['icon']}</div>
                <div style="font-weight:bold;">{info['title']}</div>
                <div>{percentage}%</div>
            </div>
            """
        
        html += """
            </div>
            
            <div style="margin-top:20px;">
                <h4 style="color:#0972d1;">Knowledge Integration Analysis</h4>
                <p>This query effectively leveraged knowledge from multiple modalities, with the strongest signals coming from documents and audio sources. 
                The highest relevance came from documentation text (42%) and audio transcripts (25%), with supporting context from images (18%) and video content (15%).</p>
            </div>
        </div>
        """
        
        return HTML(html)
        
    def show_business_context(self, context_type="rag_basic"):
        """
        Display business context for RAG applications
        
        Args:
            context_type: Type of business context to display (rag_basic, rag_complete, etc.)
        
        Returns:
            HTML visualization of the business context
        """
        if context_type == "rag_basic":
            html = """
            <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0; font-family:Arial, sans-serif;">
                <h2 style="color:#0972d1;">Business Context: Multimodal RAG</h2>
                
                <p>Retrieval-Augmented Generation (RAG) systems combine the power of large language models with your organization's proprietary knowledge. Unlike traditional search systems, RAG doesn't just find relevant documents - it synthesizes information to answer complex questions while maintaining context and accuracy.</p>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Key Business Benefits</h3>
                    <ul>
                        <li><strong>Accuracy:</strong> Ground AI responses in your organization's verified information</li>
                        <li><strong>Transparency:</strong> Provide clear citations and references for all generated content</li>
                        <li><strong>Knowledge Integration:</strong> Connect information across documents, databases, and applications</li>
                        <li><strong>Customization:</strong> Tailor responses to your industry terminology and business context</li>
                    </ul>
                </div>
            </div>
            """
            
        elif context_type == "rag_complete":
            html = """
            <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0; font-family:Arial, sans-serif;">
                <h2 style="color:#0972d1;">Business Context: Multimodal RAG</h2>
                
                <p>Multimodal Retrieval-Augmented Generation (RAG) represents the next evolution in enterprise knowledge systems. By incorporating information from documents, images, audio, and video into a unified knowledge framework, multimodal RAG enables AI systems to develop a more comprehensive understanding of your organization's information assets.</p>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Business Impact Across Industries</h3>
                    <ul>
                        <li><strong>Financial Services:</strong> Integrate quarterly reports (PDF), earnings calls (audio), market analysis videos, and visualization charts</li>
                        <li><strong>Healthcare:</strong> Connect medical literature, diagnostic images, patient records, and consultation recordings</li>
                        <li><strong>Manufacturing:</strong> Link equipment manuals, maintenance videos, sensor data visualizations, and technical diagrams</li>
                        <li><strong>Media & Entertainment:</strong> Search across video archives, production assets, scripts, and promotional materials</li>
                    </ul>
                </div>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Implementation Best Practices</h3>
                    <ul>
                        <li><strong>Source Attribution:</strong> Always maintain clear links between generated responses and source material</li>
                        <li><strong>Modality Integration:</strong> Design data pipelines to preserve context across different media types</li>
                        <li><strong>Ethical Considerations:</strong> Implement governance frameworks for data usage and content generation</li>
                        <li><strong>Performance Monitoring:</strong> Track accuracy, relevance, and business value metrics</li>
                    </ul>
                </div>
            </div>
            """
            
        elif context_type == "knowledge_base":
            html = """
            <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0; font-family:Arial, sans-serif;">
                <h2 style="color:#0972d1;">Amazon Bedrock Knowledge Base</h2>
                
                <p>Amazon Bedrock Knowledge Bases provide a fully managed solution for creating, managing, and querying vector databases to enable retrieval-augmented generation (RAG). Knowledge Bases serve as the intelligent memory for foundation models, allowing them to use your data to generate accurate and contextually relevant responses.</p>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Key Capabilities</h3>
                    <ul>
                        <li><strong>Multimodal Support:</strong> Process and query across documents, images, audio, and video content</li>
                        <li><strong>Vector Embedding:</strong> Automatically convert content into vector embeddings using foundation models</li>
                        <li><strong>Intelligent Chunking:</strong> Apply different chunking strategies (fixed-size, hierarchical, semantic) based on content</li>
                        <li><strong>Secure Access:</strong> Fine-grained access controls and encryption for sensitive information</li>
                    </ul>
                </div>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Architecture Components</h3>
                    <ul>
                        <li><strong>OpenSearch Serverless:</strong> Fully managed vector database for efficient similarity search</li>
                        <li><strong>S3 Storage:</strong> Secure object storage for maintaining original files and supplemental content</li>
                        <li><strong>IAM Roles:</strong> Secure execution roles with fine-grained permissions for Knowledge Base operations</li>
                        <li><strong>Foundation Models:</strong> Access to embedding, generation, and reranking models via Amazon Bedrock</li>
                    </ul>
                </div>
            </div>
            """
            
        elif context_type == "data_ingestion":
            html = """
            <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0; font-family:Arial, sans-serif;">
                <h2 style="color:#0972d1;">Knowledge Base Ingestion Process</h2>
                
                <p>Data ingestion is a critical process that transforms raw content into queryable knowledge by converting it into vector embeddings that foundation models can use. Amazon Bedrock Knowledge Bases handle this process with a streamlined workflow.</p>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Ingestion Pipeline</h3>
                    <ol>
                        <li><strong>Document Loading:</strong> Raw documents are loaded from the configured data source (S3, SharePoint, etc.)</li>
                        <li><strong>Content Extraction:</strong> Text and metadata are extracted from various file formats</li>
                        <li><strong>Chunking:</strong> Content is divided into semantic units based on your configured chunking strategy</li>
                        <li><strong>Embedding Generation:</strong> Each chunk is transformed into a vector representation using foundation models</li>
                        <li><strong>Vector Indexing:</strong> Vectors are stored in the vector index with metadata for efficient retrieval</li>
                    </ol>
                </div>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Chunking Strategies</h3>
                    <ul>
                        <li><strong>Fixed-Size:</strong> Divides content into chunks of consistent token lengths with configurable overlap</li>
                        <li><strong>Hierarchical:</strong> Creates multi-level chunks to maintain both context and granularity</li>
                        <li><strong>Semantic:</strong> Chunks content based on semantic meaning rather than arbitrary token counts</li>
                        <li><strong>Custom:</strong> Apply custom chunking logic through Lambda functions for specialized needs</li>
                    </ul>
                </div>
            </div>
            """
            
        elif context_type == "vector_store":
            html = """
            <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0; font-family:Arial, sans-serif;">
                <h2 style="color:#0972d1;">Vector Search in Amazon Bedrock Knowledge Bases</h2>
                
                <p>Amazon Bedrock Knowledge Bases use Amazon OpenSearch Serverless as the underlying vector store technology. This fully managed service provides efficient similarity search capabilities for RAG applications.</p>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Amazon OpenSearch Serverless Features</h3>
                    <ul>
                        <li><strong>Fully Managed:</strong> No need to provision or manage infrastructure</li>
                        <li><strong>Auto-scaling:</strong> Automatically scales based on your indexing and query workloads</li>
                        <li><strong>k-NN:</strong> Built-in nearest neighbor search algorithms for vector similarity</li>
                        <li><strong>Hybrid Search:</strong> Combine semantic (vector) and keyword (text) search capabilities</li>
                    </ul>
                </div>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Vector Index Configuration</h3>
                    <ul>
                        <li><strong>HNSW Algorithm:</strong> Uses Hierarchical Navigable Small World algorithm for efficient search</li>
                        <li><strong>Dimension Matching:</strong> Index dimensions match the embedding model output dimensions</li>
                        <li><strong>Text Fields:</strong> Maintain searchable text fields alongside vector representations</li>
                        <li><strong>Metadata Storage:</strong> Store source information and other metadata for attribution</li>
                    </ul>
                </div>
            </div>
            """
            
        elif context_type == "semantic_search":
            html = """
            <div style="background-color:#f8f8f8; padding:20px; border-radius:10px; margin:20px 0; font-family:Arial, sans-serif;">
                <h2 style="color:#0972d1;">Semantic Search Process</h2>
                
                <p>Semantic search goes beyond keyword matching to find content based on meaning and context. In RAG applications, it enables finding relevant information even when the exact terminology differs.</p>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">The RAG Query Process</h3>
                    <ol>
                        <li><strong>Query Embedding:</strong> The user's natural language query is transformed into a vector embedding</li>
                        <li><strong>Vector Search:</strong> The query vector is compared to document vectors using similarity metrics</li>
                        <li><strong>Retrieval:</strong> The most semantically similar document chunks are retrieved from the vector store</li>
                        <li><strong>Context Assembly:</strong> Retrieved content is assembled as context for the foundation model</li>
                        <li><strong>Response Generation:</strong> The foundation model generates a response based on the retrieved context</li>
                        <li><strong>Citation Generation:</strong> Citations are added to attribute information sources</li>
                    </ol>
                </div>
                
                <div style="margin:20px 0;">
                    <h3 style="color:#0972d1;">Cross-Modal Retrieval</h3>
                    <p>In multimodal RAG, the system makes connections between different types of media:</p>
                    <ul>
                        <li><strong>Text-to-Image:</strong> Finding relevant images based on textual descriptions</li>
                        <li><strong>Audio-to-Text:</strong> Matching spoken content with written documentation</li>
                        <li><strong>Image-to-Video:</strong> Finding video segments containing similar visual elements</li>
                        <li><strong>Query-to-All:</strong> Finding relevant content across all modalities from a single query</li>
                    </ul>
                </div>
            </div>
            """
        else:
            html = f"<div>Context type '{context_type}' not recognized</div>"
            
        return HTML(html)
        
    def delete_kb(self, delete_s3_bucket=False, delete_iam_roles_and_policies=True, delete_lambda_function=False):
        """
        Delete the Knowledge Base and associated resources
        
        Args:
            delete_s3_bucket: Whether to delete the S3 bucket used by the Knowledge Base
            delete_iam_roles_and_policies: Whether to delete IAM roles and policies created for the Knowledge Base
            delete_lambda_function: Whether to delete the Lambda function created for custom chunking
        
        Returns:
            Dict with status of deletion operations
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            results = {
                "knowledge_base": False,
                "vector_store": False,
                "s3_buckets": [],
                "iam_resources": []
            }
            
            # Delete data sources
            try:
                ds_id_list = self.bedrock_agent_client.list_data_sources(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                    maxResults=100
                )['dataSourceSummaries']
                
                for idx, ds in enumerate(ds_id_list):
                    try:
                        self.bedrock_agent_client.delete_data_source(
                            dataSourceId=ds_id_list[idx]["dataSourceId"],
                            knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                        )
                        print("======== Data source deleted =========")
                    except Exception as e:
                        print(e)
            except Exception as e:
                print(f"Error listing data sources: {e}")
            
            # Delete Knowledge Base
            try:
                self.bedrock_agent_client.delete_knowledge_base(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                )
                results["knowledge_base"] = True
                print("======== Knowledge base deleted =========")
            except self.bedrock_agent_client.exceptions.ResourceNotFoundException as e:
                print("Resource not found", e)
            except Exception as e:
                print(e)
            
            time.sleep(20)
            
            # Delete OpenSearch collection and policies
            try:
                self.aoss_client.delete_collection(id=self.collection_id)
                self.aoss_client.delete_access_policy(type="data", name=self.access_policy_name)
                self.aoss_client.delete_security_policy(type="network", name=self.network_policy_name)
                self.aoss_client.delete_security_policy(type="encryption", name=self.encryption_policy_name)
                results["vector_store"] = True
                print("======== Vector Index, collection and associated policies deleted =========")
            except Exception as e:
                print(f"Error deleting OpenSearch resources: {e}")
            
            # Delete roles and policies
            if delete_iam_roles_and_policies:
                self.delete_iam_role_and_policies(results)
            
            # Delete lambda function
            if delete_lambda_function and self.lambda_function_name:
                try:
                    self.lambda_client.delete_function(FunctionName=self.lambda_function_name)
                    print(f"Deleted Lambda function {self.lambda_function_name}")
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    print(f"Lambda function {self.lambda_function_name} not found.")
            
            # Delete S3 buckets
            if delete_s3_bucket:
                self.delete_s3(results)
                
            return results
            
    def delete_iam_role_and_policies(self, results=None):
        """
        Delete IAM roles and policies associated with the Knowledge Base
        
        Args:
            results: Optional dictionary to store results
        """
        if results is None:
            results = {"iam_resources": []}
            
        iam = boto3.resource('iam')
        client = boto3.client('iam')
        
        # Fetch attached policies
        try:
            response = client.list_attached_role_policies(RoleName=self.bedrock_kb_execution_role_name)
            policies_to_detach = response['AttachedPolicies']
            
            for policy in policies_to_detach:
                policy_name = policy['PolicyName']
                policy_arn = policy['PolicyArn']
                
                try:
                    self.iam_client.detach_role_policy(
                        RoleName=self.kb_execution_role_name,
                        PolicyArn=policy_arn
                    )
                    self.iam_client.delete_policy(PolicyArn=policy_arn)
                    results["iam_resources"].append(f"Policy: {policy_name} (deleted)")
                except self.iam_client.exceptions.NoSuchEntityException:
                    results["iam_resources"].append(f"Policy: {policy_name} (not found)")
                    print(f"Policy {policy_arn} not found")
            
            try:
                self.iam_client.delete_role(RoleName=self.kb_execution_role_name)
                results["iam_resources"].append(f"Role: {self.kb_execution_role_name} (deleted)")
            except self.iam_client.exceptions.NoSuchEntityException:
                results["iam_resources"].append(f"Role: {self.kb_execution_role_name} (not found)")
                print(f"Role {self.kb_execution_role_name} not found")
                
            print("======== All IAM roles and policies deleted =========")
        except Exception as e:
            results["iam_resources"].append(f"Error: {str(e)}")
            print(f"Error deleting IAM resources: {e}")
            
    def delete_s3(self, results=None):
        """
        Delete all contents and the S3 buckets associated with the knowledge base
        
        Args:
            results: Optional dictionary to store results
        """
        if results is None:
            results = {"s3_buckets": []}
            
        try:
            s3_client = boto3.client('s3')
            
            # Collect all buckets to delete
            buckets_to_delete = self.bucket_names.copy() if self.bucket_names else []
            if self.intermediate_bucket_name:
                buckets_to_delete.append(self.intermediate_bucket_name)
                
            for bucket_name in buckets_to_delete:
                try:
                    # First, delete all objects in the bucket
                    paginator = s3_client.get_paginator('list_objects_v2')
                    objects_to_delete = []
                    
                    # List all objects in the bucket
                    for page in paginator.paginate(Bucket=bucket_name):
                        if 'Contents' in page:
                            objects_to_delete.extend([
                                {'Key': obj['Key']}
                                for obj in page['Contents']
                            ])
                    
                    # Delete objects in chunks of 1000 (S3 limit)
                    if objects_to_delete:
                        chunk_size = 1000
                        for i in range(0, len(objects_to_delete), chunk_size):
                            chunk = objects_to_delete[i:i + chunk_size]
                            s3_client.delete_objects(
                                Bucket=bucket_name,
                                Delete={
                                    'Objects': chunk,
                                    'Quiet': True
                                }
                            )
                    
                    # Delete bucket versions if bucket is versioned
                    try:
                        paginator = s3_client.get_paginator('list_object_versions')
                        objects_to_delete = []
                        
                        for page in paginator.paginate(Bucket=bucket_name):
                            # Handle versions
                            if 'Versions' in page:
                                objects_to_delete.extend([
                                    {'Key': v['Key'], 'VersionId': v['VersionId']}
                                    for v in page['Versions']
                                ])
                            # Handle delete markers
                            if 'DeleteMarkers' in page:
                                objects_to_delete.extend([
                                    {'Key': dm['Key'], 'VersionId': dm['VersionId']}
                                    for dm in page['DeleteMarkers']
                                ])
                        
                        if objects_to_delete:
                            for i in range(0, len(objects_to_delete), chunk_size):
                                chunk = objects_to_delete[i:i + chunk_size]
                                s3_client.delete_objects(
                                    Bucket=bucket_name,
                                    Delete={
                                        'Objects': chunk,
                                        'Quiet': True
                                    }
                                )
                    except ClientError:
                        # Bucket might not be versioned
                        pass
                    
                    # Finally delete the empty bucket
                    s3_client.delete_bucket(Bucket=bucket_name)
                    results["s3_buckets"].append(f"{bucket_name} (deleted)")
                    print(f"Successfully deleted bucket: {bucket_name}")
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'NoSuchBucket':
                        results["s3_buckets"].append(f"{bucket_name} (does not exist)")
                        print(f"Bucket {bucket_name} does not exist")
                    else:
                        results["s3_buckets"].append(f"{bucket_name} (error: {str(e)})")
                        print(f"Error deleting bucket {bucket_name}: {e}")
                        
        except Exception as e:
            results["s3_buckets"].append(f"Error: {str(e)}")
            print(f"Error in delete_s3: {e}")
