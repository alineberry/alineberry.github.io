import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

export function readingTime(body: string): number {
  const words = body.trim().split(/\s+/).length;
  return Math.max(1, Math.round(words / 200));
}

export function postHref(p: Post): string {
  // Normalize permalink to /foo/ (trailing slash, no leading dup)
  const raw = p.data.permalink.replace(/^\/+/, '').replace(/\/+$/, '');
  return '/' + raw + '/';
}

export async function getAllPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', (p) => !p.data.draft);
  return posts.sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );
}

export async function getEssays(): Promise<Post[]> {
  return (await getAllPosts()).filter((p) => !p.data.isNote);
}

export async function getNotes(): Promise<Post[]> {
  return (await getAllPosts()).filter((p) => p.data.isNote);
}
