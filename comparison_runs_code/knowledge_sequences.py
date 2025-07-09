import torch


def generate_knowledge_sequence_vanilla_precomputed(model, options_embeddings_image, options_embedding_text, col_embedding, separator_embedding, batch_metadata, use_noise='no'):
    used_device = next(model.image_encoder.parameters()).device
    
    assert len(options_embeddings_image) == len(options_embedding_text)
    #sep_token_embedding = self.tokenizer.encode(self.tokenizer.sep_token_id)
    # Move static embeddings to the correct device once
    col_embedding = col_embedding.to(device=used_device)
    separator_embedding = separator_embedding.to(device=used_device)
    #global_embeddings_image = global_embeddings_image.to(used_device=used_device)
    
    ### move tensors to GPU
    for batch in range(len(options_embeddings_image)):
        img_dict = options_embeddings_image[batch]
        txt_dict = options_embedding_text[batch]
        for key in img_dict.keys():
            img_dict[key] = img_dict[key].to(device=used_device)
            txt_dict[key] = txt_dict[key].to(device=used_device)

    batch_knowledge_sequence = []
    batch_knowledge_ttid = []

    for batch_index in range(len(options_embedding_text)):
        img_options_embedding = options_embeddings_image[batch_index]
        text_option_embedding = options_embedding_text[batch_index]
        
        assert img_options_embedding.keys() == text_option_embedding.keys(), \
            "Mismatched keys between image and text embeddings."


        total_seq_len = 0
        for option in batch_metadata[batch_index]['options']:
            img_emb = img_options_embedding[option]
            text_emb = text_option_embedding[option]       # (num_tokens x 1 x 768)


            ### replace embeddings with 0 / noise
            if use_noise == 'yes':
                img_emb = torch.zeros(img_emb.size(0), 1, 768).to(device=used_device)
                text_emb =  torch.zeros(text_emb.size(0), 1, 768).to(device=used_device)
            
            
            total_seq_len += text_emb.size(0)
            total_seq_len += col_embedding.size(0)      # usually 1
            total_seq_len += 1 if img_emb.size(0) > 1 else img_emb.size(0) ### for l1_yes questions, we'll max later, so still only 1 token
            total_seq_len += separator_embedding.size(0) # usually 1
            

        ttid_tensor = torch.empty(total_seq_len, dtype=torch.long, device=used_device)

        pieces = []
        cursor = 0
        
        for option in batch_metadata[batch_index]['options']:
            img_emb  = img_options_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True
            text_emb = text_option_embedding[option]   # (num_tokens × 1 × 768), requires_grad=True

            ### in vanilla mode we have yes:1x1768 img embedding
            if img_emb.size(0) > 1:
                img_emb, _ = torch.max(img_emb, dim=0, keepdim=True)

            if use_noise == 'yes':
                text_emb = torch.zeros_like(text_emb).detach()
                img_emb  = torch.zeros_like(img_emb).detach()


            # ─(a) TEXT EMBEDDING──────────────────────────────────────────────────────────────
            pieces.append(text_emb)
            n_text = text_emb.size(0)
            # All text tokens share token‐type ID “5” (for example). Adjust if needed.
            ttid_tensor[cursor : cursor + n_text] = 5
            cursor += n_text
            # ─(b) COLUMN EMBEDDING────────────────────────────────────────────────────────────
            pieces.append(col_embedding)
            n_col = col_embedding.size(0)  # usually 1
            # We give col_embedding the SAME ttid as text (5), or pick your convention.
            ttid_tensor[cursor : cursor + n_col] = 5
            cursor += n_col
            # ─(c) IMAGE EMBEDDING─────────────────────────────────────────────────────────────
            pieces.append(img_emb)
            n_img = img_emb.size(0)
            # All image tokens share token‐type ID “6,” for instance.
            ttid_tensor[cursor : cursor + n_img] = 6
            cursor += n_img
            # ─(d) SEPARATOR EMBEDDING────────────────────────────────────────────────────────
            pieces.append(separator_embedding)
            n_sep = separator_embedding.size(0)  # usually 1
            ttid_tensor[cursor : cursor + n_sep] = 6
            cursor += n_sep
            
        assert cursor == total_seq_len, f"Cursor mismatch: {cursor} vs {total_seq_len}"
        
        

        # ─── (4C) Concatenate all pieces into one long tensor ─────────────────────────────────
        # Because `text_emb` and `img_emb` were not detached, they carry requires_grad=True,
        # so the resulting `batch_tensor` automatically becomes part of the autograd graph.
        batch_tensor = torch.cat(pieces, dim=0)  # shape: (total_seq_len × 1 × 768)
        # At this point: batch_tensor.requires_grad == True
        # The `.grad_fn` on batch_tensor will point back into each text_emb and img_emb.

        # ─── (4D) Append to the outputs ────────────────────────────────────────────────────────
        

        
        batch_knowledge_sequence.append(batch_tensor)
        batch_knowledge_ttid.append(ttid_tensor)
        
        pieces.clear()
        del pieces

    return batch_knowledge_sequence, batch_knowledge_ttid