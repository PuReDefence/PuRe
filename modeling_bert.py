from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, ModelOutput

from pure import PURE, StatisticsPooling


class PureBertForSequenceClassification(BertPreTrainedModel):
    """
    BERT model with PURE for sequence classification tasks.
    
    This model applies PURE transformation to BERT embeddings before classification,
    which can help improve representation quality through dimensionality reduction
    and noise filtering techniques.
    """

    def __init__(
        self, 
        config, 
        svd_rank: int = 5, 
        npc: int = 1, 
        center: bool = False, 
        num_iters: int = 1, 
        alpha: float = 1.0, 
        do_pcr: bool = True, 
        do_pfsa: bool = True, 
        label_smoothing: float = 0.0, 
    ):
        """
        Initialize the PureBertForSequenceClassification model.
        
        Args:
            config: BERT configuration object
            svd_rank: Target rank for SVD decomposition in PURE
            npc: Number of principal components for PURE
            center: Whether to center the data in PURE
            num_iters: Number of iterations for PURE processing
            alpha: Alpha parameter for PURE
            do_pcr: Whether to apply Principal Component Removal
            do_pfsa: Whether to apply Principal Feature Space Adjustment
            label_smoothing: Label smoothing factor for cross-entropy loss
        """
        super().__init__(config)
        
        # Store configuration
        self.label_smoothing = label_smoothing
        self.num_labels = config.num_labels
        self.config = config

        # Initialize BERT backbone without pooling layer
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Determine dropout probability
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        
        # Initialize PURE transformation layer
        self.pure = PURE(
            in_dim=config.hidden_size, 
            target_rank=svd_rank, 
            npc=npc, 
            center=center, 
            num_iters=num_iters, 
            alpha=alpha, 
            do_pcr=do_pcr, 
            do_pfsa=do_pfsa
        )
        
        # Pooling and classification layers
        self.mean_pooling = StatisticsPooling(return_mean=True, return_std=False)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.post_init()

    def forward_pure_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ModelOutput:
        """
        Forward pass that returns PURE-transformed embeddings without classification.
        
        Args:
            input_ids: Token ids of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            token_type_ids: Token type ids of shape (batch_size, sequence_length)
            position_ids: Position ids of shape (batch_size, sequence_length)
            head_mask: Head mask for attention layers
            inputs_embeds: Pre-computed embeddings instead of input_ids
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return ModelOutput instead of tuple
            
        Returns:
            ModelOutput containing PURE-transformed token embeddings
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get BERT embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply PURE transformation
        token_embeddings = outputs.last_hidden_state
        token_embeddings = self.pure(token_embeddings, attention_mask)

        return ModelOutput(last_hidden_state=token_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Token ids of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            token_type_ids: Token type ids of shape (batch_size, sequence_length)
            position_ids: Position ids of shape (batch_size, sequence_length)
            head_mask: Head mask for attention layers
            inputs_embeds: Pre-computed embeddings instead of input_ids
            labels: Labels for computing loss, shape (batch_size,) for classification
                   or (batch_size, num_labels) for multi-label classification
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return SequenceClassifierOutput instead of tuple
            
        Returns:
            SequenceClassifierOutput or tuple containing loss, logits, hidden_states, attentions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get BERT embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply PURE transformation and pooling
        token_embeddings = outputs.last_hidden_state
        token_embeddings = self.pure(token_embeddings, attention_mask)
        pooled_output = self.mean_pooling(token_embeddings).squeeze(1)
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Calculate loss if labels are provided
        loss = self._calculate_loss(logits, labels) if labels is not None else None

        # Return results
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate appropriate loss based on problem type.
        
        Args:
            logits: Model predictions of shape (batch_size, num_labels)
            labels: Ground truth labels
            
        Returns:
            Computed loss tensor
        """
        # Determine problem type if not set
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        # Calculate loss based on problem type
        if self.config.problem_type == "regression":
            loss_fct = nn.MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
                
        elif self.config.problem_type == "single_label_classification":
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        else:
            raise ValueError(f"Unsupported problem type: {self.config.problem_type}")

        return loss
